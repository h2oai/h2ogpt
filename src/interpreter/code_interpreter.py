import subprocess
import webbrowser
import tempfile
import threading
import traceback
import platform
import time
import ast
import sys
import os
import re


def run_html(html_content):
    # Create a temporary HTML file with the content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        f.write(html_content.encode())

    # Open the HTML file with the default web browser
    webbrowser.open('file://' + os.path.realpath(f.name))

    return f"Saved to {os.path.realpath(f.name)} and opened with the user's default web browser."


# Mapping of languages to their start, run, and print commands
language_map = {
  "python": {
    # Python is run from this interpreter with sys.executable
    # in interactive, quiet, and unbuffered mode
    "start_cmd": sys.executable + " -i -q -u",
    "print_cmd": 'print("{}")'
  },
  "R": {
    # R is run from this interpreter with R executable
    # in interactive, quiet, and unbuffered mode
    "start_cmd": "R -q --vanilla",
    "print_cmd": 'print("{}")'
  },
  "shell": {
    # On Windows, the shell start command is `cmd.exe`
    # On Unix, it should be the SHELL environment variable (defaults to 'bash' if not set)
    "start_cmd": 'cmd.exe' if platform.system() == 'Windows' else os.environ.get('SHELL', 'bash'),
    "print_cmd": 'echo "{}"'
  },
  "javascript": {
    "start_cmd": "node -i",
    "print_cmd": 'console.log("{}")'
  },
  "applescript": {
    # Starts from shell, whatever the user's preference (defaults to '/bin/zsh')
    # (We'll prepend "osascript -e" every time, not once at the start, so we want an empty shell)
    "start_cmd": os.environ.get('SHELL', '/bin/zsh'),
    "print_cmd": 'log "{}"'
  },
  "html": {
    "open_subrocess": False,
    "run_function": run_html,
  }
}

# Get forbidden_commands (disabled)
"""
with open("interpreter/forbidden_commands.json", "r") as f:
  forbidden_commands = json.load(f)
"""


class CodeInterpreter:
  """
  Code Interpreters display and run code in different languages.

  They can control code blocks on the terminal, then be executed to produce an output which will be displayed in real-time.
  """

  def __init__(self, language, debug_mode):
    self.language = language
    self.proc = None
    self.active_line = None
    self.debug_mode = debug_mode


  def update_active_block(self):
      """
      This will also truncate the output,
      which we need to do every time we update the active block.
      """
      # Strip then truncate the output if necessary
      self.output = truncate_output(self.output)

      # Display it
      self.active_block.active_line = self.active_line
      self.active_block.output = self.output
      self.active_block.refresh()

  def run(self):
    """
    Executes code.
    """

    # Get code to execute
    self.code = self.active_block.code

    # Check for forbidden commands (disabled)
    """
    for line in self.code.split("\n"):
      if line in forbidden_commands:
        message = f"This code contains a forbidden command: {line}"
        message += "\n\nPlease contact the Open Interpreter team if this is an error."
        self.active_block.output = message
        return message
    """

    # Should we keep a subprocess open? True by default
    open_subrocess = language_map[self.language].get("open_subrocess", True)

    # Start the subprocess if it hasn't been started
    if not self.proc and open_subrocess:
      try:
        self.start_process()
      except:
        # Sometimes start_process will fail!
        # Like if they don't have `node` installed or something.

        traceback_string = traceback.format_exc()
        self.output = traceback_string
        self.update_active_block()

        # Before you return, wait for the display to catch up?
        # (I'm not sure why this works)
        time.sleep(0.1)

        return self.output

    # Reset output
    self.output = ""

    # Use the print_cmd for the selected language
    self.print_cmd = language_map[self.language].get("print_cmd")
    code = self.code

    # Add print commands that tell us what the active line is
    if self.print_cmd:

        traceback_string = traceback.format_exc()
        self.output = traceback_string
        self.update_active_block()

        # Before you return, wait for the display to catch up?
        # (I'm not sure why this works)
        time.sleep(0.1)

        return self.output

    # Remove any whitespace lines, as this will break indented blocks
    # (are we sure about this? test this)
    code_lines = code.split("\n")
    code_lines = [c for c in code_lines if c.strip() != ""]
    code = "\n".join(code_lines)

    # Add end command (we'll be listening for this so we know when it ends)
    if self.print_cmd and self.language != "applescript": # Applescript is special. Needs it to be a shell command because 'return' (very common) will actually return, halt script
      code += "\n\n" + self.print_cmd.format('END_OF_EXECUTION')

    # Applescript-specific processing
    if self.language == "applescript":
      # Escape double quotes
      code = code.replace('"', r'\"')
      # Wrap in double quotes
      code = '"' + code + '"'
      # Prepend start command
      code = "osascript -e " + code
      # Append end command
      code += '\necho "END_OF_EXECUTION"'

    # Debug
    if self.debug_mode:
      print("Running code:")
      print(code)
      print("---")

    # HTML-specific processing (and running)
    if self.language == "html":
      output = language_map["html"]["run_function"](code)
      return output

    # Reset self.done so we can .wait() for it
    self.done = threading.Event()
    self.done.clear()

    # Write code to stdin of the process
    try:
      self.proc.stdin.write(code + "\n")
      self.proc.stdin.flush()
    except BrokenPipeError:
      # It can just.. break sometimes? Let's fix this better in the future
      # For now, just try again
      self.start_process()
      self.run()
      return

    # Wait until execution completes
    self.done.wait()

    # Before you return, wait for the display to catch up?
    # (I'm not sure why this works)
    time.sleep(0.1)

    # Return code output
    return self.output

  def save_and_display_stream(self, stream, is_error_stream):
    # Handle each line of output
    for line in iter(stream.readline, ''):

      if self.debug_mode:
        print("Recieved output line:")
        print(line)
        print("---")

      line = line.strip()

      # Python's interactive REPL outputs a million things
      # So we clean it up:
      if self.language == "python":
        if re.match(r'^(\s*>>>\s*|\s*\.\.\.\s*)', line):
          continue


      # Check if it's a message we added (like ACTIVE_LINE)
      # Or if we should save it to self.output
      if line.startswith("ACTIVE_LINE:"):
        self.active_line = int(line.split(":")[1])
      elif "END_OF_EXECUTION" in line:
        self.done.set()
        self.active_line = None
      elif is_error_stream and "KeyboardInterrupt" in line:
        raise KeyboardInterrupt
      else:
        self.output += "\n" + line
        self.output = self.output.strip()

      self.update_active_block()

