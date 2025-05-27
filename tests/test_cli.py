import unittest
from unittest.mock import patch
import sys
import os

# Add the project root to the Python path to allow importing key_idea_finder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from key_idea_finder import parse_arguments as parse_key_idea_finder_args
from narration import parse_arguments as parse_narration_args
from book_cover import parse_arguments as parse_book_cover_args # Import for book_cover.py

class TestKeyIdeaFinderCLI(unittest.TestCase):

    def test_parse_arguments_with_book_path_only(self):
        """Test parsing with only the required book_path argument."""
        test_argv = ['test_book.pdf']
        args = parse_key_idea_finder_args(test_argv)
        self.assertEqual(args.book_path, 'test_book.pdf')
        self.assertIsNone(args.output_file, "Output file should be None by default")

    def test_parse_arguments_with_book_path_and_output_file(self):
        """Test parsing with book_path and --output_file."""
        test_argv = ['my_book.txt', '--output_file', 'output.md']
        args = parse_key_idea_finder_args(test_argv)
        self.assertEqual(args.book_path, 'my_book.txt')
        self.assertEqual(args.output_file, 'output.md')

    def test_parse_arguments_with_book_path_and_short_output_file(self):
        """Test parsing with book_path and -o for output_file."""
        test_argv = ['another_book.pdf', '-o', 'report.txt']
        args = parse_key_idea_finder_args(test_argv)
        self.assertEqual(args.book_path, 'another_book.pdf')
        self.assertEqual(args.output_file, 'report.txt')

    @patch('sys.stderr') # Suppress argparse error messages during this test
    @patch('argparse.ArgumentParser.exit') # Patch ArgumentParser.exit
    def test_parse_arguments_missing_book_path(self, mock_argparse_exit, mock_sys_stderr):
        """Test parsing when the required book_path is missing."""
        test_argv = ['--output_file', 'out.md'] # Missing book_path
        parse_key_idea_finder_args(test_argv)
        mock_argparse_exit.assert_any_call(2, unittest.mock.ANY)

    @patch('sys.stderr') # Suppress argparse error messages
    @patch('argparse.ArgumentParser.exit') # Patch ArgumentParser.exit
    def test_parse_arguments_no_arguments(self, mock_argparse_exit, mock_sys_stderr):
        """Test parsing with no arguments provided."""
        test_argv = [] # No arguments
        parse_key_idea_finder_args(test_argv)
        mock_argparse_exit.assert_any_call(2, unittest.mock.ANY)

    @patch('sys.stdout') # Suppress help message output during test
    @patch('argparse.ArgumentParser.exit') # Patch ArgumentParser.exit
    def test_parse_arguments_help_short(self, mock_argparse_exit, mock_sys_stdout):
        """Test -h for help. Argparse's help action calls parser.exit()."""
        test_argv = ['-h']
        parse_key_idea_finder_args(test_argv)
        
        self.assertTrue(mock_argparse_exit.called, "ArgumentParser.exit was not called for -h")
        # Check if the first call was parser.exit() (recorded as empty args tuple by mock),
        # which implies status 0 due to method defaults.
        self.assertEqual(len(mock_argparse_exit.call_args_list[0][0]), 0, 
                         f"First call for -h should be exit() (empty args recorded), implying status 0. Got: {mock_argparse_exit.call_args_list[0]}")

    @patch('sys.stdout') # Suppress help message output
    @patch('argparse.ArgumentParser.exit') # Patch ArgumentParser.exit
    def test_parse_arguments_help_long(self, mock_argparse_exit, mock_sys_stdout):
        """Test --help for help. Argparse's help action calls parser.exit()."""
        test_argv = ['--help']
        parse_key_idea_finder_args(test_argv)

        self.assertTrue(mock_argparse_exit.called, "ArgumentParser.exit was not called for --help")
        # Check if the first call was parser.exit() (recorded as empty args tuple by mock),
        # which implies status 0 due to method defaults.
        self.assertEqual(len(mock_argparse_exit.call_args_list[0][0]), 0,
                         f"First call for --help should be exit() (empty args recorded), implying status 0. Got: {mock_argparse_exit.call_args_list[0]}")

if __name__ == '__main__':
    unittest.main()

class TestNarrationCLI(unittest.TestCase):

    def test_parse_arguments_book_path_only(self):
        """Test narration.py CLI with only the required book_path argument."""
        test_argv = ['sample_book.pdf']
        args = parse_narration_args(test_argv)
        self.assertEqual(args.book_path, 'sample_book.pdf')
        self.assertEqual(args.output_file, "corrected_narration_and_log.md", "Default output file should be used.")

    def test_parse_arguments_book_path_and_output_file(self):
        """Test narration.py CLI with book_path and --output_file."""
        test_argv = ['my_book.txt', '--output_file', 'custom_output.md']
        args = parse_narration_args(test_argv)
        self.assertEqual(args.book_path, 'my_book.txt')
        self.assertEqual(args.output_file, 'custom_output.md')

    def test_parse_arguments_book_path_and_short_output_file(self):
        """Test narration.py CLI with book_path and -o for output_file."""
        test_argv = ['another_book.pdf', '-o', 'narration_report.txt']
        args = parse_narration_args(test_argv)
        self.assertEqual(args.book_path, 'another_book.pdf')
        self.assertEqual(args.output_file, 'narration_report.txt')

    @patch('sys.stderr')
    @patch('argparse.ArgumentParser.exit')
    def test_parse_arguments_missing_book_path(self, mock_argparse_exit, mock_sys_stderr):
        """Test narration.py CLI when the required book_path is missing."""
        test_argv = ['-o', 'out.md'] # Missing book_path
        parse_narration_args(test_argv)
        mock_argparse_exit.assert_any_call(2, unittest.mock.ANY)

    @patch('sys.stderr')
    @patch('argparse.ArgumentParser.exit')
    def test_parse_arguments_no_arguments(self, mock_argparse_exit, mock_sys_stderr):
        """Test narration.py CLI with no arguments provided."""
        test_argv = [] # No arguments
        parse_narration_args(test_argv)
        mock_argparse_exit.assert_any_call(2, unittest.mock.ANY)

    @patch('sys.stdout')
    @patch('argparse.ArgumentParser.exit')
    def test_parse_arguments_help_short(self, mock_argparse_exit, mock_sys_stdout):
        """Test narration.py CLI -h for help."""
        test_argv = ['-h']
        parse_narration_args(test_argv)
        self.assertTrue(mock_argparse_exit.called, "ArgumentParser.exit was not called for -h")
        self.assertEqual(len(mock_argparse_exit.call_args_list[0][0]), 0,
                         f"First call for -h should be exit(), implying status 0. Got: {mock_argparse_exit.call_args_list[0]}")

    @patch('sys.stdout')
    @patch('argparse.ArgumentParser.exit')
    def test_parse_arguments_help_long(self, mock_argparse_exit, mock_sys_stdout):
        """Test narration.py CLI --help for help."""
        test_argv = ['--help']
        parse_narration_args(test_argv)
        self.assertTrue(mock_argparse_exit.called, "ArgumentParser.exit was not called for --help")
        self.assertEqual(len(mock_argparse_exit.call_args_list[0][0]), 0,
                         f"First call for --help should be exit(), implying status 0. Got: {mock_argparse_exit.call_args_list[0]}")

class TestBookCoverCLI(unittest.TestCase):

    def test_parse_arguments_required_only(self):
        """Test book_cover.py CLI with only required arguments."""
        test_argv = ['--book_name', 'My Book', '--title', 'The Title', '--author', 'The Author']
        args = parse_book_cover_args(test_argv)
        self.assertEqual(args.book_name, 'My Book')
        self.assertEqual(args.title, 'The Title')
        self.assertEqual(args.author, 'The Author')
        self.assertEqual(args.output_dir, "output", "Default output_dir should be 'output'")

    def test_parse_arguments_all_args(self):
        """Test book_cover.py CLI with all arguments, including output_dir."""
        test_argv = ['--book_name', 'My Book', '--title', 'The Title', '--author', 'The Author', '--output_dir', 'custom_images/']
        args = parse_book_cover_args(test_argv)
        self.assertEqual(args.book_name, 'My Book')
        self.assertEqual(args.title, 'The Title')
        self.assertEqual(args.author, 'The Author')
        self.assertEqual(args.output_dir, 'custom_images/')

    def test_parse_arguments_all_args_short_output(self):
        """Test book_cover.py CLI with all arguments and -d for output_dir."""
        test_argv = ['--book_name', 'My Book', '--title', 'The Title', '--author', 'The Author', '-d', 'short_output/']
        args = parse_book_cover_args(test_argv)
        self.assertEqual(args.output_dir, 'short_output/')

    @patch('sys.stderr')
    @patch('argparse.ArgumentParser.exit')
    def test_parse_arguments_missing_required(self, mock_argparse_exit, mock_sys_stderr):
        """Test book_cover.py CLI with a missing required argument (e.g., --title)."""
        test_argv = ['--book_name', 'My Book', '--author', 'The Author'] # Missing --title
        parse_book_cover_args(test_argv)
        mock_argparse_exit.assert_any_call(2, unittest.mock.ANY)

    @patch('sys.stderr')
    @patch('argparse.ArgumentParser.exit')
    def test_parse_arguments_no_arguments(self, mock_argparse_exit, mock_sys_stderr):
        """Test book_cover.py CLI with no arguments."""
        test_argv = [] # No arguments
        parse_book_cover_args(test_argv)
        mock_argparse_exit.assert_any_call(2, unittest.mock.ANY)
    
    @patch('sys.stdout')
    @patch('argparse.ArgumentParser.exit')
    def test_parse_arguments_help_short(self, mock_argparse_exit, mock_sys_stdout):
        """Test book_cover.py CLI -h for help."""
        test_argv = ['-h']
        parse_book_cover_args(test_argv)
        self.assertTrue(mock_argparse_exit.called, "ArgumentParser.exit was not called for -h")
        self.assertEqual(len(mock_argparse_exit.call_args_list[0][0]), 0,
                         f"First call for -h should be exit(), implying status 0. Got: {mock_argparse_exit.call_args_list[0]}")

    @patch('sys.stdout')
    @patch('argparse.ArgumentParser.exit')
    def test_parse_arguments_help_long(self, mock_argparse_exit, mock_sys_stdout):
        """Test book_cover.py CLI --help for help."""
        test_argv = ['--help']
        parse_book_cover_args(test_argv)
        self.assertTrue(mock_argparse_exit.called, "ArgumentParser.exit was not called for --help")
        self.assertEqual(len(mock_argparse_exit.call_args_list[0][0]), 0,
                         f"First call for --help should be exit(), implying status 0. Got: {mock_argparse_exit.call_args_list[0]}")
