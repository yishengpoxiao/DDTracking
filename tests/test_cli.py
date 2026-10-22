import contextlib
import io
import unittest

from tractography import main


class CommandLineTests(unittest.TestCase):
    def test_requires_config_and_exactly_one_mode(self):
        for arguments in ([], ['--track'], ['--config', 'missing.yaml'],
                          ['--config', 'missing.yaml', '--train', '--track']):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_status:
                    main(arguments)
                self.assertEqual(exit_status.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
