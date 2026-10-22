"""Run Slicer with --additional-module-paths slicer/DDTracking and --python-script tests/slicer_smoke.py."""

import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import qt
import slicer
import vtk


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def wait_for(predicate, description, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        slicer.app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for " + description)


def make_vtp(path, empty=False):
    points = vtk.vtkPoints()
    if empty:
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(polyData)
        if writer.Write() != 1:
            raise AssertionError("Could not write empty test VTP")
        return
    points.InsertNextPoint(1.25, 2.5, 3.75)
    points.InsertNextPoint(4.5, 5.25, 6.0)
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, 0)
    line.GetPointIds().SetId(1, 1)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(line)
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polyData)
    if writer.Write() != 1:
        raise AssertionError("Could not write test VTP")


def make_nifti(path, name):
    node = slicer.util.addVolumeFromArray(np.ones((3, 3, 3), dtype=np.float32), name=name)
    try:
        if not slicer.util.saveNode(node, path):
            raise AssertionError("Could not write test NIfTI: " + path)
    finally:
        slicer.mrmlScene.RemoveNode(node)


def write_result(path, nativeVtp, mniVtp, nativeFa, registeredFa, template, streamlineCount=1):
    result = {
        "native_vtp": nativeVtp,
        "mni_vtp": mniVtp,
        "native_fa": nativeFa,
        "registered_fa": registeredFa,
        "template": template,
        "native_to_mni_transform": "",
        "streamline_count": streamlineCount,
    }
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(result, stream)


def create_helper(path):
    source = """import sys, time
mode, result_path = sys.argv[1:3]
marker = 'DDTRACKING_RESULT=' + result_path + '\\n'
for part in (marker[:12], marker[12:24], marker[24:]):
    sys.stdout.write(part)
    sys.stdout.flush()
    time.sleep(0.12)
if mode == 'wait':
    time.sleep(10)
"""
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(source)


def test_itk_transform_convention(directory):
    path = os.path.join(directory, "simple-itk-translation.tfm")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("""#Insight Transform File V1.0
# Transform 0
Transform: AffineTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 10 20 30
FixedParameters: 0 0 0
""")
    existingIDs = {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLTransformNode")}
    if not slicer.util.loadTransform(path):
        raise AssertionError("Slicer could not load the ITK transform")
    node = next(node for node in slicer.util.getNodesByClass("vtkMRMLTransformNode")
                if node.GetID() not in existingIDs)
    matrix = vtk.vtkMatrix4x4()
    node.GetMatrixTransformToParent(matrix)
    actual = np.array([[matrix.GetElement(row, column) for column in range(4)] for row in range(4)])
    expected = np.eye(4)
    expected[:3, 3] = (10, 20, -30)
    slicer.mrmlScene.RemoveNode(node)
    assert np.allclose(actual, expected), "ITK transform did not load as the expected RAS parent transform: %r" % actual


def set_path(edit, value):
    edit.setText(value)


def run_smoke_test():
    standalonePython = os.environ.get("DDTRACKING_TEST_PYTHON")
    if standalonePython:
        python = os.path.abspath(standalonePython)
    else:
        environment = slicer.util.startupEnvironment()
        searchPath = environment.get("PATH", "")
        python = (shutil.which("python3", path=searchPath)
                  or shutil.which("python", path=searchPath))
    if not python or not os.path.isfile(python):
        raise AssertionError(
            "A standalone Python executable is required; set DDTRACKING_TEST_PYTHON "
            "or add python3/python to Slicer's startup PATH"
        )
    backend = os.path.join(REPO_ROOT, "slicer_backend", "run.py")
    if not os.path.isfile(backend):
        raise AssertionError("Run this test from a source checkout containing slicer_backend/run.py")

    if getattr(slicer.modules, "ddtracking", None) is None:
        raise AssertionError(
            "DDTracking was not discovered. Add slicer/DDTracking to Additional module paths "
            "before launching this smoke test."
        )

    print("Slicer version: %s" % slicer.app.applicationVersion)
    print("Slicer Python: %s" % sys.version.split()[0])
    print("Qt version: %s" % qt.qVersion())
    print("VTK version: %s" % vtk.vtkVersion().GetVTKVersion())
    print("Standalone Python: %s" % python)

    temporary = tempfile.mkdtemp(prefix="ddtracking-slicer-smoke-")
    testDirectory = os.path.join(temporary, "paths containing spaces")
    os.makedirs(testDirectory)
    settingsPrefix = "DDTrackingSmoke/%s-%s/" % (os.getpid(), int(time.time() * 1000000))
    initialModelIDs = {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLModelNode")}
    initialVolumeIDs = {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")}
    widget = slicer.modules.ddtracking.widgetRepresentation().self()
    previousSettingsPrefix = widget.SETTINGS_PREFIX
    widget.SETTINGS_PREFIX = settingsPrefix

    try:
        widget.loadMniOutputs.checked = True
        test_itk_transform_convention(testDirectory)
        inputFiles = {}
        for name in ("model.pt", "dwi.nii.gz", "dwi.bval", "dwi.bvec", "brain.nii.gz", "template.nii.gz"):
            path = os.path.join(testDirectory, name)
            with open(path, "wb") as stream:
                stream.write(b"test")
            inputFiles[name] = path
        set_path(widget.pythonPath, python)
        set_path(widget.modelPath, inputFiles["model.pt"])
        set_path(widget.dwiPath, inputFiles["dwi.nii.gz"])
        set_path(widget.bvalPath, inputFiles["dwi.bval"])
        set_path(widget.bvecPath, inputFiles["dwi.bvec"])
        set_path(widget.brainMaskPath, inputFiles["brain.nii.gz"])
        set_path(widget.templatePath, inputFiles["template.nii.gz"])
        set_path(widget.outputPath, testDirectory)
        for edit in (widget.wmMaskPath, widget.faPath, widget.trackingConfigPath):
            edit.clear()

        nativeVtp = os.path.join(testDirectory, "native.vtp")
        mniVtp = os.path.join(testDirectory, "mni.vtp")
        nativeFa = os.path.join(testDirectory, "native-fa.nii.gz")
        registeredFa = os.path.join(testDirectory, "registered-fa.nii.gz")
        template = os.path.join(testDirectory, "mni-template.nii.gz")
        for path in (nativeVtp, mniVtp):
            make_vtp(path)
        for path, name in ((nativeFa, "native FA"), (registeredFa, "registered FA"), (template, "template")):
            make_nifti(path, name)

        firstResult = os.path.join(testDirectory, "first-result.json")
        cancelledResult = os.path.join(testDirectory, "cancelled-result.json")
        finalResult = os.path.join(testDirectory, "final-result.json")
        write_result(firstResult, nativeVtp, mniVtp, nativeFa, registeredFa, template)
        write_result(cancelledResult, nativeVtp, mniVtp, nativeFa, registeredFa, template)
        write_result(finalResult, nativeVtp, mniVtp, nativeFa, registeredFa, template)
        helper = os.path.join(testDirectory, "helper.py")
        create_helper(helper)

        widget._arguments = lambda preprocessOnly=False: [helper, "success", firstResult]
        widget.onRun()
        wait_for(lambda: widget.process is None, "chunked result marker completion")
        assert widget.statusLabel.text == "Completed", widget.statusLabel.text + "\n" + widget.log.toPlainText()
        assert widget.resultPath == firstResult, widget.resultPath
        nativeModels = [node for node in slicer.util.getNodesByClass("vtkMRMLModelNode")
                        if node.GetName() == "DDTracking native tractogram"]
        assert nativeModels, "Native tractogram was not loaded"
        point = nativeModels[-1].GetPolyData().GetPoint(0)
        assert np.allclose(point, (1.25, 2.5, 3.75)), point
        assert not nativeModels[-1].GetDisplayNode().GetVisibility(), "Native model should be hidden in MNI QC mode"
        mniModels = [node for node in slicer.util.getNodesByClass("vtkMRMLModelNode")
                     if node.GetName() == "DDTracking MNI tractogram"]
        assert mniModels and mniModels[-1].GetDisplayNode().GetVisibility(), "MNI model should be visible in QC mode"

        widget._arguments = lambda preprocessOnly=False: [helper, "wait", cancelledResult]
        widget.onRun()
        wait_for(lambda: widget.resultPath == cancelledResult, "first run marker before cancellation")
        widget.onCancel()
        wait_for(lambda: widget.process is None, "cancelled process shutdown")
        assert widget.statusLabel.text == "Cancelled", widget.statusLabel.text

        widget._arguments = lambda preprocessOnly=False: [helper, "success", finalResult]
        widget.onRun()
        wait_for(lambda: widget.process is None, "restart after cancellation")
        assert widget.statusLabel.text == "Completed", widget.statusLabel.text
        assert widget.resultPath == finalResult, widget.resultPath
        assert not widget.stdoutBuffers, "Output buffers from finished processes should be released"

        emptyNative = os.path.join(temporary, "empty-native.vtp")
        emptyMni = os.path.join(temporary, "empty-mni.vtp")
        emptyResult = os.path.join(temporary, "empty-result.json")
        make_vtp(emptyNative, empty=True)
        make_vtp(emptyMni, empty=True)
        write_result(emptyResult, emptyNative, emptyMni, nativeFa, registeredFa, template, streamlineCount=0)
        modelIDs = {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLModelNode")}
        widget._arguments = lambda preprocessOnly=False: [helper, "success", emptyResult]
        widget.onRun()
        wait_for(lambda: widget.process is None, "zero-streamline result")
        assert widget.statusLabel.text == "Completed (no streamlines)", widget.statusLabel.text
        assert modelIDs == {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLModelNode")}, "Empty tractograms should not be loaded as model nodes"
    finally:
        widget.cleanup()
        qt.QSettings().remove(settingsPrefix.rstrip("/"))
        widget.SETTINGS_PREFIX = previousSettingsPrefix
        for nodeClass, existingIDs in (("vtkMRMLModelNode", initialModelIDs),
                                       ("vtkMRMLScalarVolumeNode", initialVolumeIDs)):
            for node in slicer.util.getNodesByClass(nodeClass):
                if node.GetID() not in existingIDs:
                    slicer.mrmlScene.RemoveNode(node)
        slicer.app.processEvents()
        shutil.rmtree(temporary, ignore_errors=True)


try:
    run_smoke_test()
except Exception:
    traceback.print_exc()
    print("DDTracking Slicer smoke test failed")
    slicer.util.exit(1)
else:
    print("DDTracking Slicer smoke test passed")
    slicer.util.exit(0)
