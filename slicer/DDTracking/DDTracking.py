"""3D Slicer interface for running DDTracking outside Slicer's Python."""

import json
import os
import slicer
import qt
import ctk
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleTest,
)


class DDTracking(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "DDTracking"
        parent.categories = ["Diffusion"]
        parent.dependencies = []
        parent.contributors = ["DDTracking"]
        parent.helpText = (
            "Run DDTracking tractography with an external Python environment. "
            "FA computation and registration are performed by the pipeline."
        )
        parent.acknowledgementText = ""
        self.parent = parent


class DDTrackingWidget(ScriptedLoadableModuleWidget):
    SETTINGS_PREFIX = "DDTracking/"
    RESULT_MARKER = "DDTRACKING_RESULT="

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.process = None
        self.processTimer = None
        self.cancelRequested = False
        self.resultPath = None
        self.stdoutBuffers = {}
        self.runMode = "tracking"

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        form = qt.QFormLayout()
        self.layout.addLayout(form)

        moduleDir = os.path.dirname(os.path.abspath(__file__))
        repoRoot = os.path.abspath(os.path.join(moduleDir, "..", ".."))
        templateCandidates = (os.path.join(repoRoot, "resources", "MNI_FA_template.nii.gz"),
                              os.path.join(moduleDir, "resources", "MNI_FA_template.nii.gz"))
        defaultTemplate = next((path for path in templateCandidates if os.path.isfile(path)), templateCandidates[0])
        self.pythonPath = self._pathRow(form, "External Python", "python", "Python executable (*)")
        self.modelPath = self._pathRow(form, "Model checkpoint", "file", "Model checkpoints (*.pt *.pth);;All files (*)")
        self.dwiPath = self._pathRow(form, "DWI image", "file", "NIfTI images (*.nii *.nii.gz)")
        self.bvalPath = self._pathRow(form, "b-values", "file", "b-values (*.bval *.txt);;All files (*)")
        self.bvecPath = self._pathRow(form, "b-vectors", "file", "b-vectors (*.bvec *.txt);;All files (*)")
        self.brainMaskPath = self._pathRow(form, "Brain mask", "file", "NIfTI images (*.nii *.nii.gz)")
        self.wmMaskPath = self._pathRow(form, "White matter mask (optional)", "file", "NIfTI images (*.nii *.nii.gz)")
        self.faPath = self._pathRow(form, "Native FA (optional)", "file", "NIfTI images (*.nii *.nii.gz)")
        self.templatePath = self._pathRow(form, "MNI FA template", "file", "NIfTI images (*.nii *.nii.gz)")
        self.templatePath.setText(defaultTemplate)
        self.outputPath = self._pathRow(form, "Output folder", "directory", "")

        self.advanced = ctk.ctkCollapsibleButton()
        self.advanced.text = "Tracking options"
        self.advanced.collapsed = True
        self.layout.addWidget(self.advanced)
        advancedLayout = qt.QFormLayout(self.advanced)
        self.trackingConfigPath = self._pathRow(advancedLayout, "Tracking config (optional)", "file", "YAML files (*.yaml *.yml);;All files (*)")
        self.shell = self._spin(0, 10000, 1000, integer=True)
        self.shellTolerance = self._spin(0, 10000, 100, integer=True)
        self.seeds = self._spin(1, 1000, 1, integer=True)
        self.stepSize = self._spin(0.01, 20.0, 0.5, decimals=2)
        self.maxAngle = self._spin(0.1, 180.0, 45.0, decimals=1)
        self.minLength = self._spin(0.0, 10000.0, 40.0, decimals=1)
        self.maxLength = self._spin(0.1, 10000.0, 200.0, decimals=1)
        self.batchSize = self._spin(1, 1000000, 1024, integer=True)
        self.device = qt.QComboBox()
        self.device.addItems(["cpu", "cuda:0"])
        self.device.setEditable(True)
        for label, widget in (
            ("b-shell (s/mm²)", self.shell), ("Shell tolerance", self.shellTolerance),
            ("Seeds per voxel", self.seeds), ("Step size (voxel)", self.stepSize),
            ("Maximum angle (degrees)", self.maxAngle), ("Minimum length (mm)", self.minLength),
            ("Maximum length (mm)", self.maxLength), ("Batch size", self.batchSize),
            ("Device", self.device),
        ):
            advancedLayout.addRow(label, widget)

        self.loadMniOutputs = qt.QCheckBox("Load registered FA and MNI tractogram for QC")
        self.loadMniOutputs.checked = False
        self.layout.addWidget(self.loadMniOutputs)

        buttonLayout = qt.QHBoxLayout()
        self.layout.addLayout(buttonLayout)
        self.runButton = qt.QPushButton("Run tracking")
        self.preprocessButton = qt.QPushButton("Register FA only")
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.enabled = False
        buttonLayout.addWidget(self.runButton)
        buttonLayout.addWidget(self.preprocessButton)
        buttonLayout.addWidget(self.cancelButton)
        self.statusLabel = qt.QLabel("Ready")
        self.layout.addWidget(self.statusLabel)
        self.log = qt.QPlainTextEdit()
        self.log.readOnly = True
        self.log.setMinimumHeight(160)
        self.log.document().setMaximumBlockCount(3000)
        self.layout.addWidget(self.log)
        self.layout.addStretch(1)

        self.runButton.connect("clicked(bool)", self.onRun)
        self.preprocessButton.connect("clicked(bool)", self.onPreprocess)
        self.cancelButton.connect("clicked(bool)", self.onCancel)
        self._restoreSettings()
        self._setActive(False)

    def _pathRow(self, form, label, mode, fileFilter):
        row = qt.QWidget()
        rowLayout = qt.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        edit = qt.QLineEdit()
        edit.setClearButtonEnabled(True)
        browse = qt.QPushButton("Browse…")
        rowLayout.addWidget(edit, 1)
        rowLayout.addWidget(browse)
        form.addRow(label, row)
        browse.connect("clicked(bool)", lambda checked=False: self._browse(edit, mode, fileFilter))
        return edit

    def _browse(self, edit, mode, fileFilter):
        current = edit.text.strip()
        if mode == "directory":
            path = qt.QFileDialog.getExistingDirectory(self.parent, "Choose output folder", current or os.path.expanduser("~"))
        else:
            path = qt.QFileDialog.getOpenFileName(self.parent, "Choose file", current or os.path.expanduser("~"), fileFilter)
            if isinstance(path, tuple):
                path = path[0]
        if path:
            edit.setText(path)

    @staticmethod
    def _spin(minimum, maximum, value, integer=False, decimals=1):
        widget = qt.QSpinBox() if integer else qt.QDoubleSpinBox()
        widget.setRange(int(minimum), int(maximum)) if integer else widget.setRange(minimum, maximum)
        if not integer:
            widget.setDecimals(decimals)
        widget.value = value
        return widget

    def _restoreSettings(self):
        settings = qt.QSettings()
        for key, edit in (("python", self.pythonPath), ("model", self.modelPath), ("trackingConfig", self.trackingConfigPath), ("dwi", self.dwiPath),
                          ("bval", self.bvalPath), ("bvec", self.bvecPath), ("brainMask", self.brainMaskPath),
                          ("wmMask", self.wmMaskPath), ("fa", self.faPath), ("template", self.templatePath),
                          ("output", self.outputPath)):
            value = settings.value(self.SETTINGS_PREFIX + key)
            if value:
                edit.setText(value)

    def _saveSettings(self):
        settings = qt.QSettings()
        for key, edit in (("python", self.pythonPath), ("model", self.modelPath), ("trackingConfig", self.trackingConfigPath), ("dwi", self.dwiPath),
                          ("bval", self.bvalPath), ("bvec", self.bvecPath), ("brainMask", self.brainMaskPath),
                          ("wmMask", self.wmMaskPath), ("fa", self.faPath), ("template", self.templatePath),
                          ("output", self.outputPath)):
            settings.setValue(self.SETTINGS_PREFIX + key, edit.text.strip())

    def _setActive(self, active):
        self.runButton.enabled = not active
        self.preprocessButton.enabled = not active
        self.cancelButton.enabled = active

    def _requiredPaths(self, preprocessOnly=False):
        fields = [("External Python", self.pythonPath)]
        if not preprocessOnly:
            fields.append(("Model checkpoint", self.modelPath))
        fields.extend([
                  ("DWI image", self.dwiPath), ("b-values", self.bvalPath), ("b-vectors", self.bvecPath),
                  ("Brain mask", self.brainMaskPath), ("MNI FA template", self.templatePath),
                  ("Output folder", self.outputPath)])
        missing = [label for label, edit in fields if not edit.text.strip()]
        if missing:
            raise ValueError("Choose: " + ", ".join(missing))
        for label, edit in fields[:-1]:
            if not os.path.isfile(edit.text.strip()):
                raise ValueError(f"{label} does not exist: {edit.text.strip()}")
        if not os.access(self.pythonPath.text.strip(), os.X_OK):
            raise ValueError("External Python is not executable")
        for label, edit in (("Tracking config", self.trackingConfigPath), ("White matter mask", self.wmMaskPath),
                            ("Native FA", self.faPath)):
            if edit.text.strip() and not os.path.isfile(edit.text.strip()):
                raise ValueError(f"{label} does not exist: {edit.text.strip()}")
        if not os.path.isdir(self.outputPath.text.strip()):
            raise ValueError("Output folder must already exist")
        backend = self._backendPath()
        if not os.path.isfile(backend):
            raise ValueError("DDTracking backend is missing: " + backend)

    def _backendPath(self):
        moduleDir = os.path.dirname(os.path.abspath(__file__))
        repoRoot = os.path.abspath(os.path.join(moduleDir, "..", ".."))
        backendCandidates = (os.path.join(repoRoot, "slicer_backend", "run.py"),
                             os.path.join(moduleDir, "slicer_backend", "run.py"))
        return next((path for path in backendCandidates if os.path.isfile(path)), backendCandidates[0])

    def _arguments(self, preprocessOnly=False):
        args = [self._backendPath()]
        args.extend(["--dwi", self.dwiPath.text.strip(), "--bvals", self.bvalPath.text.strip(),
                     "--bvecs", self.bvecPath.text.strip(), "--brain-mask", self.brainMaskPath.text.strip()])
        optional = (("--wm-mask", self.wmMaskPath), ("--fa", self.faPath))
        for flag, edit in optional:
            if edit.text.strip():
                args.extend([flag, edit.text.strip()])
        if self.trackingConfigPath.text.strip():
            args.extend(["--tracking-config", self.trackingConfigPath.text.strip()])
        args.extend(["--template", self.templatePath.text.strip(),
                     "--output-dir", self.outputPath.text.strip(), "--shell", str(self.shell.value),
                     "--shell-tolerance", str(self.shellTolerance.value), "--seeds", str(self.seeds.value),
                     "--step-size", str(self.stepSize.value), "--max-angle", str(self.maxAngle.value),
                     "--min-length", str(self.minLength.value), "--max-length", str(self.maxLength.value),
                     "--batch-size", str(self.batchSize.value), "--device", self.device.currentText.strip()])
        if not preprocessOnly:
            args.extend(["--model", self.modelPath.text.strip()])
        else:
            args.append("--preprocess-only")
        return args

    def _processEnvironment(self):
        environment = qt.QProcessEnvironment()
        for key, value in slicer.util.startupEnvironment().items():
            environment.insert(str(key), str(value))
        environment.insert("PYTHONUNBUFFERED", "1")
        return environment

    def onRun(self):
        self._startRun(preprocessOnly=False)

    def onPreprocess(self):
        self._startRun(preprocessOnly=True)

    def _startRun(self, preprocessOnly):
        try:
            self._requiredPaths(preprocessOnly=preprocessOnly)
        except ValueError as error:
            slicer.util.errorDisplay(str(error), windowTitle="DDTracking")
            return
        self._saveSettings()
        self.cancelRequested = False
        self.resultPath = None
        self.runMode = "preprocess" if preprocessOnly else "tracking"
        process = qt.QProcess(self.parent)
        self.process = process
        self.stdoutBuffers[id(process)] = ""
        arguments = self._arguments(preprocessOnly=preprocessOnly)
        program = os.path.abspath(self.pythonPath.text.strip())
        process.setProcessEnvironment(self._processEnvironment())
        process.setProgram(program)
        process.setArguments(arguments)
        process.setWorkingDirectory(os.path.dirname(os.path.dirname(arguments[0])))
        process.connect("readyReadStandardOutput()", lambda p=process: self._readStdout(p))
        process.connect("readyReadStandardError()", lambda p=process: self._readStderr(p))
        process.connect("finished(int, QProcess::ExitStatus)", lambda code, status, p=process: self._finished(p, code, status))
        process.connect("errorOccurred(QProcess::ProcessError)", lambda error, p=process: self._processError(p, error))
        self.log.appendPlainText("Starting external Python: " + program)
        self.log.appendPlainText("Arguments: " + repr(arguments))
        self.statusLabel.text = "Starting…"
        self._setActive(True)
        process.start()

    def _readStdout(self, process):
        text = self._decode(process.readAllStandardOutput())
        if not text:
            return
        if process is not self.process:
            return
        bufferKey = id(process)
        output = self.stdoutBuffers.get(bufferKey, "") + text
        lines = output.split("\n")
        self.stdoutBuffers[bufferKey] = lines.pop()
        for line in lines:
            self._consumeLine(line.rstrip("\r"), "", process)

    def _readStderr(self, process):
        text = self._decode(process.readAllStandardError())
        if text and process is self.process:
            for line in text.splitlines():
                self._consumeLine(line, "[stderr] ")

    @staticmethod
    def _decode(data):
        raw = data.data() if hasattr(data, "data") else data
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def _consumeLine(self, line, prefix, process=None):
        self.log.appendPlainText(prefix + line)
        if not prefix and process is self.process and line.startswith(self.RESULT_MARKER):
            self.resultPath = line[len(self.RESULT_MARKER):].strip()

    def _processError(self, process, error):
        self.log.appendPlainText("Process error: " + str(error))
        if error == qt.QProcess.FailedToStart:
            if process is not self.process:
                return
            self._setActive(False)
            self.statusLabel.text = "Failed to start external Python"
            self.process = None
            self.stdoutBuffers.pop(id(process), None)
            self._stopProcessTimer()
            if process:
                process.deleteLater()

    def _finished(self, process, exitCode, exitStatus):
        self._readStdout(process)
        self._readStderr(process)
        output = self.stdoutBuffers.pop(id(process), "")
        if output:
            self._consumeLine(output.rstrip("\r"), "", process)
        if process is not self.process:
            process.deleteLater()
            return
        self._stopProcessTimer()
        self._setActive(False)
        self.process = None
        if process:
            process.deleteLater()
        if self.cancelRequested:
            self.statusLabel.text = "Cancelled"
            return
        if exitStatus != qt.QProcess.NormalExit or exitCode != 0:
            self.statusLabel.text = f"Failed (exit code {exitCode})"
            return
        if not self.resultPath or not os.path.isfile(self.resultPath):
            self.statusLabel.text = "Finished, but no readable result was reported"
            return
        try:
            streamlineCount = self._loadResults(self.resultPath, preprocessOnly=self.runMode == "preprocess")
            self.statusLabel.text = "Completed (no streamlines)" if streamlineCount == 0 else "Completed"
        except Exception as error:
            self.statusLabel.text = "Processing completed; output loading failed"
            slicer.util.errorDisplay(str(error), windowTitle="DDTracking output")

    def _loadResults(self, resultPath, preprocessOnly=False):
        with open(resultPath, "r", encoding="utf-8") as stream:
            result = json.load(stream)
        for existing in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            if existing.GetAttribute("DDTracking.CoordinateSystem") == "RAS":
                existing.GetDisplayNode().SetVisibility(False)
        nativeFa = slicer.util.loadVolume(result["native_fa"])
        nativeFa.SetName("DDTracking native FA")
        streamlineCount = result.get("streamline_count")
        if preprocessOnly or self.loadMniOutputs.checked:
            registeredFa = slicer.util.loadVolume(result["registered_fa"])
            registeredFa.SetName("DDTracking registered FA")
            template = slicer.util.loadVolume(result["template"])
            template.SetName("DDTracking MNI FA template")
            slicer.util.setSliceViewerLayers(background=template, foreground=registeredFa, foregroundOpacity=0.5)
        else:
            slicer.util.setSliceViewerLayers(background=nativeFa, foreground=None)
        if not preprocessOnly and streamlineCount != 0:
            nativeModel = self._loadModelRAS(result["native_vtp"], "DDTracking native tractogram")
        if not preprocessOnly and self.loadMniOutputs.checked and streamlineCount != 0:
            mniModel = self._loadModelRAS(result["mni_vtp"], "DDTracking MNI tractogram")
            nativeModel.GetDisplayNode().SetVisibility(False)
            mniModel.GetDisplayNode().SetVisibility(True)
        if streamlineCount == 0:
            self.log.appendPlainText("Tracking completed with no streamlines; FA registration outputs were loaded.")
        self.log.appendPlainText("Loaded outputs from " + os.path.abspath(resultPath))
        return streamlineCount

    @staticmethod
    def _loadModelRAS(path, name):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        if reader.GetErrorCode() != vtk.vtkErrorCode.NoError or reader.GetOutput().GetNumberOfPoints() == 0:
            raise RuntimeError("Could not read VTP file: " + path)
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        model.SetAndObservePolyData(reader.GetOutput())
        model.CreateDefaultDisplayNodes()
        model.GetDisplayNode().SetColor(0.95, 0.55, 0.15)
        model.GetDisplayNode().SetLineWidth(1.5)
        model.SetAttribute("DDTracking.CoordinateSystem", "RAS")
        return model

    def onCancel(self):
        if not self.process:
            return
        self.cancelRequested = True
        self.cancelButton.enabled = False
        self.statusLabel.text = "Cancelling…"
        self.log.appendPlainText("Cancellation requested")
        process = self.process
        process.terminate()
        self.processTimer = qt.QTimer(self.parent)
        self.processTimer.setSingleShot(True)
        self.processTimer.connect("timeout()", lambda p=process: self._killProcess(p))
        self.processTimer.start(3000)

    def _killProcess(self, process):
        if process is self.process and process.state() != qt.QProcess.NotRunning:
            process.kill()

    def _stopProcessTimer(self):
        if self.processTimer:
            self.processTimer.stop()
            self.processTimer.deleteLater()
            self.processTimer = None

    def cleanup(self):
        process = self.process
        self.process = None
        if process:
            self.stdoutBuffers.pop(id(process), None)
        self._stopProcessTimer()
        if process and process.state() != qt.QProcess.NotRunning:
            process.terminate()
            if not process.waitForFinished(3000):
                process.kill()
                process.waitForFinished(1000)
        if process:
            process.deleteLater()


class DDTrackingTest(ScriptedLoadableModuleTest):
    """Small Slicer-side smoke test for constructing the module widget."""

    def runTest(self):
        self.setUp()
        self.testWidgetSetup()

    def testWidgetSetup(self):
        parent = qt.QWidget()
        parent.setLayout(qt.QVBoxLayout())
        widget = DDTrackingWidget(parent)
        widget.setup()
        assert widget.runButton.text == "Run tracking"
        assert widget.preprocessButton.text == "Register FA only"
        assert widget.brainMaskPath is not None
        assert widget.trackingConfigPath is not None
        assert widget.templatePath.text.endswith("MNI_FA_template.nii.gz")
        widget.cleanup()
        parent.deleteLater()
