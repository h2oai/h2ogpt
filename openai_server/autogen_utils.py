import asyncio
import copy
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import typing
import warnings
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union
from types import SimpleNamespace
import uuid

from autogen.code_utils import PYTHON_VARIANTS, WIN32, _cmd, TIMEOUT_MSG, decide_use_docker, \
    check_can_use_docker_or_throw, content_str
from autogen.coding import LocalCommandLineCodeExecutor, CodeBlock, CodeExecutorFactory
from autogen.coding.base import CommandLineCodeResult
from autogen import ConversableAgent, Agent, OpenAIWrapper
from autogen import GroupChatManager
import backoff

from autogen.coding.func_with_reqs import (
    FunctionWithRequirements,
    FunctionWithRequirementsStr,
)
from autogen.coding.utils import silence_pip
from autogen.io import IOStream
from autogen.runtime_logging import logging_enabled, log_new_agent
from pydantic import Field
from termcolor import colored

from typing_extensions import ParamSpec

A = ParamSpec("A")

from openai_server.autogen_streaming import iostream_generator
from openai_server.backend_utils import convert_gen_kwargs
from openai_server.agent_utils import in_pycharm, set_python_path, extract_agent_tool

verbose = os.getenv('VERBOSE', '0').lower() == '1'

danger_mark = 'Potentially dangerous operation detected'
bad_output_mark = 'Output contains sensitive information'


class H2OCodeBlock(CodeBlock):
    """(Experimental) A class that represents a code block."""

    execute: bool = Field(description="Whether to execute the code.")


class H2OLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    def __init__(
            self,
            timeout: int = 60,
            virtual_env_context: Optional[SimpleNamespace] = None,
            work_dir: Union[Path, str] = Path("."),
            functions: List[
                Union[FunctionWithRequirements[Any, A], Callable[..., Any], FunctionWithRequirementsStr]] = [],
            functions_module: str = "functions",
            execution_policies: Optional[Dict[str, bool]] = None,
            autogen_code_restrictions_level: int = 2,
            stream_output: bool = True,
            agent_tools_usage_hard_limits: Dict[str, int] = {},
            agent_tools_usage_soft_limits: Dict[str, int] = {},
            max_stream_length: int = 4096,
            max_memory_usage: Optional[int] = 16 * 1024 ** 3,  # 16GB
    ):
        super().__init__(timeout, virtual_env_context, work_dir, functions, functions_module, execution_policies)
        self.autogen_code_restrictions_level = autogen_code_restrictions_level
        self.stream_output = stream_output
        self.agent_tools_usage_hard_limits = agent_tools_usage_hard_limits
        self.agent_tools_usage_soft_limits = agent_tools_usage_soft_limits
        self.agent_tools_usage = {}
        self.max_stream_length = max_stream_length
        self.max_memory_usage = max_memory_usage
        self.turns = 0  # for tracking

        self.filename_patterns: List[re.Pattern] = [
            re.compile(r"^<!--\s*filename:\s*([\w.-/]+)\s*-->$"),
            re.compile(r"^/\*\s*filename:\s*([\w.-/]+)\s*\*/$"),
            re.compile(r"^//\s*filename:\s*([\w.-/]+)\s*$"),
            re.compile(r"^#\s*filename:\s*([\w.-/]+)\s*$"),
        ]

    @staticmethod
    def remove_comments_strings(code: str, lang: str) -> str:
        if verbose:
            print(f"Original code:\n{code}", file=sys.stderr)

        if lang in ["bash", "shell", "sh"]:
            # Remove single-line comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            # Remove string literals (this is a simplification and might not catch all cases)
            code = re.sub(r'"[^"]*"', '', code)
            code = re.sub(r"'[^']*'", '', code)
        elif lang == "python":
            # Remove single-line comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            # Remove multi-line strings and docstrings
            code = re.sub(r'"{3}[\s\S]*?"{3}', '', code)
            code = re.sub(r"'{3}[\s\S]*?'{3}", '', code)
            # Remove string literals (this is a simplification and might not catch all cases)
            code = re.sub(r'"[^"]*"', '', code)
            code = re.sub(r"'[^']*'", '', code)

        cleaned_code = code.strip()  # Added strip() to remove leading/trailing whitespace
        if verbose:
            print(f"Cleaned code:\n{cleaned_code}", file=sys.stderr)
        return cleaned_code

    @staticmethod
    def sanitize_command(lang: str, code: str) -> None:
        shell_patterns: typing.Dict[str, str] = {
            r"\brm\b": "Deleting files or directories is not allowed.",
            r"\brm\s+-rf\b": "Use of 'rm -rf' command is not allowed.",
            r"\bmv\b.*?/dev/null": "Moving files to /dev/null is not allowed.",
            r"\bdd\b": "Use of 'dd' command is not allowed.",
            r">\s*/dev/sd[a-z][1-9]?": "Overwriting disk blocks directly is not allowed.",
            r":\(\)\{.*?\}:": "Fork bombs are not allowed.",
            r"\bsudo\b": "Use of 'sudo' command is not allowed.",
            r"\bsu\b": "Use of 'su' command is not allowed.",
            r"\bchmod\b": "Changing file permissions is not allowed.",
            r"\bchown\b": "Changing file ownership is not allowed.",
            r"\bnc\b.*?-e": "Use of netcat in command execution mode is not allowed.",
            r"\bcurl\b.*?\|\s*bash": "Piping curl output to bash is not allowed.",
            r"\bwget\b.*?\|\s*bash": "Piping wget output to bash is not allowed.",
            r"\b(systemctl|service)\s+(start|stop|restart)": "Starting, stopping, or restarting services is not allowed.",
            r"\bnohup\b": "Use of 'nohup' command is not allowed.",
            r"&\s*$": "Running commands in the background is not allowed.",
            r"\bkill\b": "Use of 'kill' command is not allowed.",
            r"\bpkill\b": "Use of 'pkill' command is not allowed.",
            r"\b(python|python3|php|node|ruby)\s+-m\s+http\.server": "Starting an HTTP server is not allowed.",
            r"\biptables\b": "Modifying firewall rules is not allowed.",
            r"\bufw\b": "Modifying firewall rules is not allowed.",
            r"\bexport\b": "Exporting environment variables is not allowed.",
            r"\benv\b": "Accessing or modifying environment variables is not allowed.",
            r"\becho\b.*?>\s*/etc/": "Writing to system configuration files is not allowed.",
            r"\bsed\b.*?-i": "In-place file editing with sed is not allowed.",
            r"\bawk\b.*?-i": "In-place file editing with awk is not allowed.",
            r"\bcrontab\b": "Modifying cron jobs is not allowed.",
            r"\bat\b": "Scheduling tasks with 'at' is not allowed.",
            r"\b(shutdown|reboot|init\s+6|telinit\s+6)\b": "System shutdown or reboot commands are not allowed.",
            r"\b(apt-get|yum|dnf|pacman)\b": "Use of package managers is not allowed.",
            r"\$\(.*?\)": "Command substitution is not allowed.",
            r"`.*?`": "Command substitution is not allowed.",
        }

        python_patterns: typing.Dict[str, str] = {
            # Deleting files or directories
            r"\bos\.(remove|unlink|rmdir)\s*\(": "Deleting files or directories is not allowed.",
            r"\bshutil\.rmtree\s*\(": "Deleting directory trees is not allowed.",

            # System and subprocess usage
            r"\bos\.system\s*\(": "Use of os.system() is not allowed.",
            r"\bsubprocess\.(run|Popen|call|check_output)\s*\(": "Use of subprocess module is not allowed.",

            # Dangerous functions
            r"\bexec\s*\(": "Use of exec() is not allowed.",
            r"\beval\s*\(": "Use of eval() is not allowed.",
            r"\b__import__\s*\(": "Use of __import__() is not allowed.",

            # Import and usage of specific modules
            r"\bimport\s+smtplib\b": "Importing smtplib (for sending emails) is not allowed.",
            r"\bfrom\s+smtplib\s+import\b": "Importing from smtplib (for sending emails) is not allowed.",

            r"\bimport\s+ctypes\b": "Importing ctypes module is not allowed.",
            r"\bfrom\s+ctypes\b": "Importing ctypes module is not allowed.",
            r"\bctypes\.\w+": "Use of ctypes module is not allowed.",

            r"\bimport\s+pty\b": "Importing pty module is not allowed.",
            r"\bpty\.\w+": "Use of pty module is not allowed.",

            r"\bplatform\.\w+": "Use of platform module is not allowed.",

            # Exiting and process management
            r"\bsys\.exit\s*\(": "Use of sys.exit() is not allowed.",
            r"\bos\.chmod\s*\(": "Changing file permissions is not allowed.",
            r"\bos\.chown\s*\(": "Changing file ownership is not allowed.",
            r"\bos\.setuid\s*\(": "Changing process UID is not allowed.",
            r"\bos\.setgid\s*\(": "Changing process GID is not allowed.",
            r"\bos\.fork\s*\(": "Forking processes is not allowed.",

            # Scheduler, debugger, pickle, and marshall usage
            r"\bsched\.\w+": "Use of sched module (for scheduling) is not allowed.",
            r"\bcommands\.\w+": "Use of commands module is not allowed.",
            r"\bpdb\.\w+": "Use of pdb (debugger) is not allowed.",
            r"\bpickle\.loads\s*\(": "Use of pickle.loads() is not allowed.",
            r"\bmarshall\.loads\s*\(": "Use of marshall.loads() is not allowed.",

            # HTTP server usage
            r"\bhttp\.server\b": "Running HTTP servers is not allowed.",
        }

        # patterns can always block if appear in code
        any_patterns = ['H2OGPT_MODEL_LOCK', 'H2OGPT_MAIN_KWARGS', 'H2OGPT_FUNCTION_API_KEY',
                        'H2OGPT_FUNCTION_PORT', 'H2OGPT_SSL_KEYFILE_PASSWORD', 'H2OGPT_AUTH', 'H2OGPT_AUTH_FILENAME',
                        'H2OGPT_ENFORCE_H2OGPT_API_KEY', 'H2OGPT_ENFORCE_H2OGPT_UI_KEY',
                        'H2OGPT_H2OGPT_API_KEYS', 'H2OGPT_KEY', 'GRADIO_H2OGPT_H2OGPT_KEY',
                        'H2OGPT_H2OGPT_KEY',
                        ]

        if os.getenv('STRICT_KEY_USAGE', '0') == '1':
            # allow broader patterns if user wants to be stricter, so no insertion of keys into chat and usage of keys
            any_patterns += ['REPLICATE_API_TOKEN',
                             'ANTHROPIC_API_KEY', 'AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY',
                             'GOOGLE_API_KEY', 'TWILIO_AUTH_TOKEN', 'OPENAI_AZURE_KEY',
                             'PINECONE_API_KEY', 'GROQ_SECRET_ACCESS_KEY', 'OPENAI_APY_KEY',
                             'ELEVENLABS_API_KEY', 'PINECONE_ENV', 'GROQ_API_KEY', 'OPENAI_AZURE_KEY',
                             'HUGGINGFACE_API_TOKEN',
                             'MISTRAL_API_KEY', 'OPENAI_API_KEY',
                             ]
        # Do NOT include these as just patterns, since used by tools:
        # just shown for reference to avoid being added later:
        used_by_tools = ['H2OGPT_OPENAI_API_KEY', 'S2_API_KEY,' 'NEWS_API_KEY', 'SERPAPI_API_KEY',
                         'WOLFRAM_ALPHA_APPID', 'STT_OPENAI_API_KEY', 'IMAGEGEN_OPENAI_API_KEY']
        assert used_by_tools

        patterns = shell_patterns if lang in ["bash", "shell", "sh"] else python_patterns
        combined_pattern = "|".join(f"(?P<pat{i}>{pat})" for i, pat in enumerate(patterns.keys()))
        combined_pattern = re.compile(combined_pattern, re.MULTILINE | re.IGNORECASE)

        # Remove comments and strings before checking patterns
        cleaned_code = H2OLocalCommandLineCodeExecutor.remove_comments_strings(code, lang)

        match = re.search(combined_pattern, cleaned_code)
        if match:
            for i, pattern in enumerate(patterns.keys()):
                if match.group(f"pat{i}"):
                    raise ValueError(f"{danger_mark}: {patterns[pattern]}\n\n{cleaned_code}")

        if any(any_pattern in code for any_pattern in any_patterns):
            raise ValueError(f"{danger_mark}: {any_patterns}\n\n{cleaned_code}")

    def _get_file_name_from_content(self, code: str, workspace_path: Path) -> Optional[str]:
        lines = code.split("\n")
        for line in lines:
            line = line.strip()
            for pattern in self.filename_patterns:
                matches = pattern.match(line)
                if matches is not None:
                    filename = matches.group(1).strip()

                    # Validate filename
                    if not re.match(r'^[\w.-/]+$', filename):
                        continue  # Invalid filename, try next match

                    # Construct the path
                    path = Path(filename)

                    # Convert workspace_path to an absolute path at the start
                    workspace_path = workspace_path.resolve()

                    # Ensure the path doesn't try to go outside the workspace
                    try:
                        resolved_path = workspace_path.joinpath(path).resolve()
                        if resolved_path.is_relative_to(workspace_path):
                            return str(resolved_path)
                    except ValueError:
                        # Path would be outside the workspace, skip it
                        continue

        return None

    def __execute_code_dont_check_setup(self, code_blocks: List[CodeBlock]) -> CommandLineCodeResult:
        # nearly identical to parent, but with control over guardrails via self.sanitize_command
        logs_all = ""
        file_names = []
        exitcode = -2
        for code_block in code_blocks:
            lang, code = code_block.language, code_block.code

            # DETERMINE LANGUAGE
            lang = lang.lower()

            # GET FILENAME and adjust LANGUAGE
            try:
                # Check if there is a filename comment
                filename = self._get_file_name_from_content(code, self._work_dir)
                # override filename and lang if tool use is detected
                cwd = os.path.abspath(os.getcwd())
                if filename and \
                        code_block.execute and \
                        f'python {cwd}/openai_server/agent_tools/' in code and \
                        filename.endswith('.py'):
                    # switch back to shell if was wrongly .py extension
                    code_block.language = lang = 'shell'
                    filename = filename.replace('.py', '.sh')
                # override lang if filename is detected, less error-prone than using code block lang
                elif filename and filename.endswith('.sh'):
                    code_block.language = lang = 'shell'
                elif filename and filename.endswith('.py'):
                    code_block.language = lang = 'python'
            except ValueError:
                return CommandLineCodeResult(exit_code=1, output="Filename is not in the workspace")

            if self.autogen_code_restrictions_level >= 2:
                self.sanitize_command(lang, code)
            elif self.autogen_code_restrictions_level == 1:
                LocalCommandLineCodeExecutor.sanitize_command(lang, code)
            code = silence_pip(code, lang)

            if lang in PYTHON_VARIANTS:
                lang = "python"

            if WIN32 and lang in ["sh", "shell"]:
                lang = "ps1"

            if lang not in self.SUPPORTED_LANGUAGES:
                # In case the language is not supported, we return an error message.
                exitcode = 1
                logs_all += "\n" + f"unknown language {lang}"
                break

            execute_code = self.execution_policies.get(lang, False)

            if filename is None:
                # create a file with an automatically generated name
                code_hash = md5(code.encode()).hexdigest()
                filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"
            written_file = (self._work_dir / filename).resolve()
            with written_file.open("w", encoding="utf-8") as f:
                f.write(code)
            file_names.append(written_file)

            if not execute_code or hasattr(code_block, 'execute') and not code_block.execute:
                # Just return a message that the file is saved.
                logs_all += f"Code saved to {str(written_file)}\n"
                exitcode = 0
                continue

            program = _cmd(lang)
            cmd = [program, str(written_file.absolute())]
            env = os.environ.copy()

            if self._virtual_env_context:
                virtual_env_abs_path = os.path.abspath(self._virtual_env_context.bin_path)
                path_with_virtualenv = rf"{virtual_env_abs_path}{os.pathsep}{env['PATH']}"
                env["PATH"] = path_with_virtualenv
                if WIN32:
                    activation_script = os.path.join(virtual_env_abs_path, "activate.bat")
                    cmd = [activation_script, "&&", *cmd]

            try:
                if self.stream_output:
                    if 'src' not in sys.path:
                        sys.path.append('src')
                    from src.utils import execute_cmd_stream
                    exec_func = execute_cmd_stream
                else:
                    exec_func = subprocess.run
                from autogen.io import IOStream
                iostream = IOStream.get_default()
                result = exec_func(
                    cmd, cwd=self._work_dir, capture_output=True, text=True,
                    timeout=float(self._timeout), env=env,
                    print_func=iostream.print,
                    guard_func=functools.partial(H2OLocalCommandLineCodeExecutor.text_guardrail, any_fail=False),
                    max_stream_length=self.max_stream_length,
                    max_memory_usage=self.max_memory_usage,
                )
                iostream.print("\n\n**Completed execution of code block.**\n\nENDOFTURN\n")
            except subprocess.TimeoutExpired:
                logs_all += "\n" + TIMEOUT_MSG
                # Same exit code as the timeout command on linux.
                exitcode = 124
                break

            logs_all += result.stderr
            logs_all += result.stdout
            exitcode = result.returncode

            if exitcode != 0:
                break

        code_file = str(file_names[0]) if len(file_names) > 0 else None
        self.turns += 1
        return CommandLineCodeResult(exit_code=exitcode, output=logs_all, code_file=code_file)

    @staticmethod
    def is_in_container() -> bool:
        # Is this Python running in a container (Docker, Kubelet)
        try:
            with open("/proc/self/cgroup", "r") as f:
                for l in f.readlines():
                    if "docker" in l or "kubepods" in l:
                        return True
        except FileNotFoundError:
            pass
        return False

    def _execute_code_dont_check_setup(self, code_blocks: List[CodeBlock]) -> CommandLineCodeResult:
        multiple_executable_code_detected = False
        try:
            # skip code blocks with # execution: false
            code_blocks_len0 = len(code_blocks)

            code_blocks_new = []
            for code_block in code_blocks:
                if '# execution: false' not in code_block.code and \
                        '# execution:' in code_block.code in code_block.code:
                    code_block_new = H2OCodeBlock(code=code_block.code, language=code_block.language, execute=True)
                else:
                    code_block_new = H2OCodeBlock(code=code_block.code, language=code_block.language, execute=False)
                code_blocks_new.append(code_block_new)
            code_blocks = code_blocks_new
            code_blocks_exec = [x for x in code_blocks if x.execute]
            # Executable code block limitation
            if len(code_blocks_exec) > 1:
                multiple_executable_code_detected = True
                code_blocks_exec = code_blocks_exec[:1]

            code_blocks_no_exec = [x for x in code_blocks if not x.execute]

            # ensure no plots pop-up if in pycharm mode or outside docker
            if not self.is_in_container():
                for code_block in code_blocks_exec:
                    lang, code = code_block.language, code_block.code
                    if lang == 'python':
                        code_block.code = """
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
""" + code_block.code
                # merge back
                code_blocks = code_blocks_exec + code_blocks_no_exec

            # Update agent tool usage if there is any
            self.update_agent_tool_usages(code_blocks_exec)
            ret = self.__execute_code_dont_check_setup(code_blocks)

            if ret.exit_code == -2 or len(code_blocks_exec) == 0 and code_blocks_len0 > 0:
                ret = CommandLineCodeResult(exit_code=0,
                                            output="""
<no_code_executed_notes>
* Code block present, but no code executed (execution tag was false or not present for all code blocks).
* This is expected if you had code blocks but they were not meant for python or shell execution.
* For example, you may have shown code for demonstration purposes.
* If you intended to execute code, be sure to add the comment: # execution: true and try again.
* If no code execution was expected, do not respond or react to this "no_code_execution" text and instead directly and immediately provide the actual answer to the user's original question. You can repeat your non-executable code mentioned in your previous message if that's what the user is looking for.
</no_code_executed_notes>
""")
        except Exception as e:
            if danger_mark in str(e):
                print(f"Code Danger Error: {e}\n\n{code_blocks}", file=sys.stderr)
                # dont' fail, just return the error so LLM can adjust
                ret = CommandLineCodeResult(exit_code=1, output=str(e))
            else:
                raise
        try:
            ret = self.output_guardrail(ret)
        except Exception as e:
            if bad_output_mark in str(e):
                print(f"Code Output Danger Error: {e}\n\n{code_blocks}\n\n{ret}", file=sys.stderr)
                # dont' fail, just return the error so LLM can adjust
                ret = CommandLineCodeResult(exit_code=1, output=str(e))
            else:
                raise

        # Truncate output if it is too long
        ret = self.truncate_output(ret)
        # Add executed code note if needed
        ret = self.executed_code_note(ret, multiple_executable_code_detected)
        ret = self.agent_tool_usage_note(ret)
        return ret

    def update_agent_tool_usages(self, code_blocks: List[CodeBlock]) -> None:
        any_update = False
        for code_block in code_blocks:
            agent_tool = extract_agent_tool(code_block.code)
            if agent_tool:
                agent_tool = os.path.basename(agent_tool).replace('.py', '')
                if agent_tool not in self.agent_tools_usage:
                    any_update = True
                    self.agent_tools_usage[agent_tool] = 1
                else:
                    any_update = True
                    self.agent_tools_usage[agent_tool] += 1
        if any_update:
            print(f"Step {self.turns} has agent tool usage: {self.agent_tools_usage}")

    @staticmethod
    def executed_code_note(ret: CommandLineCodeResult,
                           multiple_executable_code_detected: bool = False) -> CommandLineCodeResult:
        if ret.exit_code == 0:
            if multiple_executable_code_detected:
                executable_code_limitation_warning = """
* Code execution is limited to running one code block at a time, that's why only the first code block was executed.
* You must have only one executable code block at a time in your message.
"""
            else:
                executable_code_limitation_warning = ""
            if executable_code_limitation_warning:
                ret.output += f"""
<code_executed_notes>
{executable_code_limitation_warning}
</code_executed_notes>
"""
        return ret

    def agent_tool_usage_note(self, ret) -> CommandLineCodeResult:
        for k, v in self.agent_tools_usage.items():
            # could make hard limit strictly hard, but this should help for now
            if k in self.agent_tools_usage_hard_limits and self.agent_tools_usage_hard_limits[k] < v:
                ret.output += f"""\n<agent_tool_usage_note>
Error: You have used the agent tool "{k}" more than {v} times in this conversation.  You MUST stop using it.
</agent_tool_usage_note>
"""
            elif k in self.agent_tools_usage_soft_limits and self.agent_tools_usage_soft_limits[k] < v:
                ret.output += f"""\n<agent_tool_usage_note>
Warning: You have used the agent tool "{k}" more than {v} times in this conversation. Please use it judiciously.
</agent_tool_usage_note>
"""
        return ret

    @staticmethod
    def output_guardrail(ret: CommandLineCodeResult) -> CommandLineCodeResult:
        ret.output = H2OLocalCommandLineCodeExecutor.text_guardrail(ret.output)
        return ret

    @staticmethod
    def text_guardrail(text, any_fail=False, max_bad_lines=3, just_filter_out=True):
        # List of API key environment variable names to check
        api_key_names = ['OPENAI_AZURE_KEY', 'OPENAI_AZURE_API_BASE',
                         'TWILIO_AUTH_TOKEN', 'NEWS_API_KEY', 'OPENAI_API_KEY_JON',
                         'H2OGPT_H2OGPT_KEY', 'TWITTER_API_KEY', 'FACEBOOK_ACCESS_TOKEN', 'API_KEY', 'LINKEDIN_API_KEY',
                         'STRIPE_API_KEY', 'ADMIN_PASS', 'S2_API_KEY', 'ANTHROPIC_API_KEY', 'AUTH_TOKEN',
                         'AWS_SERVER_PUBLIC_KEY', 'OPENAI_API_KEY', 'HUGGING_FACE_HUB_TOKEN', 'AWS_ACCESS_KEY_ID',
                         'SERPAPI_API_KEY', 'WOLFRAM_ALPHA_APPID', 'AWS_SECRET_ACCESS_KEY', 'ACCESS_TOKEN',
                         'SLACK_API_TOKEN', 'MISTRAL_API_KEY', 'TOGETHERAI_API_TOKEN', 'GITHUB_TOKEN', 'SECRET_KEY',
                         'GOOGLE_API_KEY', 'REPLICATE_API_TOKEN', 'GOOGLE_CLIENT_SECRET', 'GROQ_API_KEY',
                         'AWS_SERVER_SECRET_KEY', 'H2OGPT_OPENAI_BASE_URL', 'H2OGPT_OPENAI_API_KEY',
                         'GRADIO_H2OGPT_H2OGPT_KEY', 'IMAGEGEN_OPENAI_BASE_URL',
                         'IMAGEGEN_OPENAI_API_KEY',
                         'STT_OPENAI_BASE_URL', 'STT_OPENAI_API_KEY',
                         'H2OGPT_MODEL_LOCK', 'PINECONE_API_KEY', 'TEST_SERVER', 'INVOCATION_ID', 'ELEVENLABS_API_KEY',
                         'HUGGINGFACE_API_TOKEN', 'PINECONE_ENV', 'PINECONE_API_SECRET',
                         'GROQ_SECRET_ACCESS_KEY', 'BING_API_KEY',
                         ]

        # Get the values of these environment variables
        set_api_key_names = set(api_key_names)
        api_key_dict = {key: os.getenv(key, '') for key in set_api_key_names if os.getenv(key, '')}
        set_api_key_values = set(list(api_key_dict.values()))

        # Expanded set of allowed (dummy) values
        set_allowed = {
            '', 'EMPTY', 'DUMMY', 'null', 'NULL', 'Null',
            'YOUR_API_KEY', 'YOUR-API-KEY', 'your-api-key', 'your_api_key',
            'ENTER_YOUR_API_KEY_HERE', 'INSERT_API_KEY_HERE',
            'API_KEY_GOES_HERE', 'REPLACE_WITH_YOUR_API_KEY',
            'PLACEHOLDER', 'EXAMPLE_KEY', 'TEST_KEY', 'SAMPLE_KEY',
            'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            '0000000000000000000000000000000000000000',
            '1111111111111111111111111111111111111111',
            'abcdefghijklmnopqrstuvwxyz123456',
            '123456789abcdefghijklmnopqrstuvwxyz',
            'sk_test_', 'pk_test_',  # Common prefixes for test keys
            'MY_SECRET_KEY', 'MY_API_KEY', 'MY_AUTH_TOKEN',
            'CHANGE_ME', 'REPLACE_ME', 'YOUR_TOKEN_HERE',
            'N/A', 'NA', 'None', 'not_set', 'NOT_SET', 'NOT-SET',
            'undefined', 'UNDEFINED', 'foo', 'bar',
            'https://api.openai.com', 'https://api.openai.com/v1',
            'https://api.gpt.h2o.ai/v1', 'http://0.0.0.0:5000/v1',
            'https://h2ogpt.openai.azure.com/',
            # Add any other common dummy values you've encountered
        }
        set_allowed = {x.lower() for x in set_allowed}

        # Filter out allowed (dummy) values
        api_key_values = [value.lower() for value in set_api_key_values if value and value.lower() not in set_allowed]

        if text:
            api_key_values = sorted(filter(bool, api_key_values), key=len, reverse=True)

            # Compile a regex pattern outside the loop
            pattern = '|'.join(map(re.escape, api_key_values))
            regex = re.compile(pattern)

            bad_lines = 0
            bad_lines_text = []
            # try to remove offending lines first, if only 1-2 lines, then maybe logging and not code itself
            lines = []
            for line in text.split('\n'):
                if any(api_key_value in line.lower() for api_key_value in api_key_values):
                    bad_lines += 1
                    bad_lines_text.append(line)
                    if just_filter_out:
                        print(f"Sensitive information found in output, so removed text: {line}")

                        # Use the compiled regex to replace all api_key_values at once
                        line = regex.sub('', line)
                        # for api_key_value in api_key_values:
                        #    line = line.replace(api_key_value, '')
                        lines.append(line)
                    else:
                        print(f"Sensitive information found in output, so removed line: {line}")
                        # e.g. H2OGPT_OPENAI_BASE_URL can appear from logging events from httpx
                        continue
                else:
                    lines.append(line)
            text = '\n'.join(lines)

            bad_msg = f"{bad_output_mark}.  Attempt to access sensitive information has been detected and reported as a violation."
            if bad_lines >= max_bad_lines or bad_lines > 0 and any_fail:
                print("\nBad Output:\n", text)
                print("\nbad_lines_text:\n", bad_lines_text)
                raise ValueError(bad_msg)

            # Check if any API key value is in the output and collect all violations
            violated_keys = []
            violated_values = []
            api_key_dict_reversed = {v: k for k, v in api_key_dict.items()}
            for api_key_value in api_key_values:
                if api_key_value in text.lower():
                    # Find the corresponding key name(s) for the violated value
                    violated_key = api_key_dict_reversed[api_key_value]
                    violated_keys.append(violated_key)
                    violated_values.append(api_key_value)

            # If any violations were found, raise an error with all violated keys
            if violated_keys:
                error_message = f"Output contains sensitive information. Violated keys: {', '.join(violated_keys)}"
                print(error_message)
                print("\nBad Output:\n", text)
                print(
                    f"Output contains sensitive information. Violated keys: {', '.join(violated_keys)}\n Violated values: {', '.join(violated_values)}")
                raise ValueError(bad_msg)

        return text

    @staticmethod
    def truncate_output(ret: CommandLineCodeResult) -> CommandLineCodeResult:
        if ret.exit_code == 1:
            # then failure, truncated more
            max_output_length = 2048  # about 512 tokens
        else:
            max_output_length = 10000  # about 2500 tokens

        # can't be sure if need head or tail more in general, so split in half
        head_length = max_output_length // 2

        if len(ret.output) > max_output_length:
            trunc_message = f"\n\n...\n\n"
            tail_length = max_output_length - head_length - len(trunc_message)
            head_part = ret.output[:head_length]
            headless_part = ret.output[head_length:]
            tail_part = headless_part[-tail_length:]
            truncated_output = (
                    head_part +
                    trunc_message +
                    tail_part
            )
            ret.output = truncated_output

        return ret


error_patterns = [
    r"Rate limit reached",
    r"Connection timeout",
    r"Server unavailable",
    r"Internal server error",
    r"incomplete chunked read",
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backoff")


def backoff_handler(details):
    logger.info(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries. Exception: {details['exception']}")


class H2OConversableAgent(ConversableAgent):
    @backoff.on_exception(backoff.expo,
                          Exception,
                          max_tries=5,
                          giveup=lambda e: not any(re.search(pattern, str(e)) for pattern in error_patterns),
                          on_backoff=backoff_handler)
    # init is same, but with ConversableAgent replaced with H2OConversableAgent since they didn't organize class well
    def __init__(
            self,
            name: str,
            system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
            is_termination_msg: Optional[Callable[[Dict], bool]] = None,
            max_consecutive_auto_reply: Optional[int] = None,
            human_input_mode: typing.Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
            function_map: Optional[Dict[str, Callable]] = None,
            code_execution_config: Union[Dict, typing.Literal[False]] = False,
            llm_config: Optional[Union[Dict, typing.Literal[False]]] = None,
            default_auto_reply: Union[str, Dict] = "",
            description: Optional[str] = None,
            chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
            # below only matter if code_execution_config is set
            max_turns: Optional[int] = None,
            initial_confidence_level: Optional[int] = 0,
    ):
        self.max_turns = max_turns
        self.turns = 0
        self._confidence_level = initial_confidence_level

        code_execution_config = (
            code_execution_config.copy() if hasattr(code_execution_config, "copy") else code_execution_config
        )

        self._name = name
        # a dictionary of conversations, default value is list
        if chat_messages is None:
            self._oai_messages = defaultdict(list)
        else:
            self._oai_messages = chat_messages

        self._oai_system_message = [{"content": system_message, "role": "system"}]
        self._description = description if description is not None else system_message
        self._is_termination_msg = (
            is_termination_msg
            if is_termination_msg is not None
            else (lambda x: content_str(x.get("content")) == "TERMINATE")
        )
        # Take a copy to avoid modifying the given dict
        if isinstance(llm_config, dict):
            try:
                llm_config = copy.deepcopy(llm_config)
            except TypeError as e:
                raise TypeError(
                    "Please implement __deepcopy__ method for each value class in llm_config to support deepcopy."
                    " Refer to the docs for more details: https://microsoft.github.io/autogen/docs/topics/llm_configuration#adding-http-client-in-llm_config-for-proxy"
                ) from e

        self._validate_llm_config(llm_config)

        if logging_enabled():
            log_new_agent(self, locals())

        # Initialize standalone client cache object.
        self.client_cache = None

        self.human_input_mode = human_input_mode
        self._max_consecutive_auto_reply = (
            max_consecutive_auto_reply if max_consecutive_auto_reply is not None else self.MAX_CONSECUTIVE_AUTO_REPLY
        )
        self._consecutive_auto_reply_counter = defaultdict(int)
        self._max_consecutive_auto_reply_dict = defaultdict(self.max_consecutive_auto_reply)
        self._function_map = (
            {}
            if function_map is None
            else {name: callable for name, callable in function_map.items() if self._assert_valid_name(name)}
        )
        self._default_auto_reply = default_auto_reply
        self._reply_func_list = []
        self._human_input = []
        self.reply_at_receive = defaultdict(bool)
        self.register_reply([Agent, None], H2OConversableAgent.generate_oai_reply)
        self.register_reply([Agent, None], H2OConversableAgent.a_generate_oai_reply, ignore_async_in_sync_chat=True)

        # Setting up code execution.
        # Do not register code execution reply if code execution is disabled.
        if code_execution_config is not False:
            # If code_execution_config is None, set it to an empty dict.
            if code_execution_config is None:
                warnings.warn(
                    "Using None to signal a default code_execution_config is deprecated. "
                    "Use {} to use default or False to disable code execution.",
                    stacklevel=2,
                )
                code_execution_config = {}
            if not isinstance(code_execution_config, dict):
                raise ValueError("code_execution_config must be a dict or False.")

            # We have got a valid code_execution_config.
            self._code_execution_config = code_execution_config

            if self._code_execution_config.get("executor") is not None:
                if "use_docker" in self._code_execution_config:
                    raise ValueError(
                        "'use_docker' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    )

                if "work_dir" in self._code_execution_config:
                    raise ValueError(
                        "'work_dir' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    )

                if "timeout" in self._code_execution_config:
                    raise ValueError(
                        "'timeout' in code_execution_config is not valid when 'executor' is set. Use the appropriate arg in the chosen executor instead."
                    )

                # Use the new code executor.
                self._code_executor = CodeExecutorFactory.create(self._code_execution_config)
                self.register_reply([Agent, None], H2OConversableAgent._generate_code_execution_reply_using_executor)
            else:
                # Legacy code execution using code_utils.
                use_docker = self._code_execution_config.get("use_docker", None)
                use_docker = decide_use_docker(use_docker)
                check_can_use_docker_or_throw(use_docker)
                self._code_execution_config["use_docker"] = use_docker
                self.register_reply([Agent, None], H2OConversableAgent.generate_code_execution_reply)
        else:
            # Code execution is disabled.
            self._code_execution_config = False

        self.register_reply([Agent, None], H2OConversableAgent.generate_tool_calls_reply)
        self.register_reply([Agent, None], H2OConversableAgent.a_generate_tool_calls_reply,
                            ignore_async_in_sync_chat=True)
        self.register_reply([Agent, None], H2OConversableAgent.generate_function_call_reply)
        self.register_reply(
            [Agent, None], H2OConversableAgent.a_generate_function_call_reply, ignore_async_in_sync_chat=True
        )
        self.register_reply([Agent, None], H2OConversableAgent.check_termination_and_human_reply)
        self.register_reply(
            [Agent, None], H2OConversableAgent.a_check_termination_and_human_reply, ignore_async_in_sync_chat=True
        )

        # Registered hooks are kept in lists, indexed by hookable method, to be called in their order of registration.
        # New hookable methods should be added to this list as required to support new agent capabilities.
        self.hook_lists: Dict[str, List[Callable]] = {
            "process_last_received_message": [],
            "process_all_messages_before_reply": [],
            "process_message_before_send": [],
        }

    def _generate_oai_reply_from_client(self, llm_client, messages, cache) -> typing.Union[str, typing.Dict, None]:
        try:
            return super()._generate_oai_reply_from_client(llm_client, messages, cache)
        except Exception as e:
            if any(re.search(pattern, str(e)) for pattern in error_patterns):
                logger.info(f"Encountered retryable error: {str(e)}")
                raise  # Re-raise the exception to trigger backoff
            else:
                logger.error(f"Encountered non-retryable error: {str(e)}")
                raise  # If it doesn't match our patterns, raise the original exception

    def generate_oai_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> typing.Tuple[bool, Union[str, Dict, None]]:
        valid, extracted_response = super().generate_oai_reply(messages, sender, config)
        if isinstance(extracted_response, str) and 'ENDOFTURN' not in extracted_response:
            delta = '\n\nENDOFTURN\n'
            from autogen.io import IOStream
            iostream = IOStream.get_default()
            iostream.print(delta)
            extracted_response += delta
        return (False, None) if extracted_response is None else (True, extracted_response)

    def _generate_code_execution_reply_using_executor(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Union[Dict, typing.Literal[False]]] = None,
    ):
        valid, output = self.__generate_code_execution_reply_using_executor(messages, sender, config)
        if output and 'ENDOFTURN' not in output:
            delta = '\n\nENDOFTURN\n'
            from autogen.io import IOStream
            iostream = IOStream.get_default()
            iostream.print(delta)
            output += delta
        self.turns += 1
        return valid, output

    def __generate_code_execution_reply_using_executor(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Union[Dict, typing.Literal[False]]] = None,
    ):
        """Generate a reply using code executor."""
        iostream = IOStream.get_default()

        if config is not None:
            raise ValueError("config is not supported for _generate_code_execution_reply_using_executor.")
        if self._code_execution_config is False:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        last_n_messages = self._code_execution_config.get("last_n_messages", "auto")

        if not (isinstance(last_n_messages, (int, float)) and last_n_messages >= 0) and last_n_messages != "auto":
            raise ValueError("last_n_messages must be either a non-negative integer, or the string 'auto'.")

        num_messages_to_scan = last_n_messages
        if last_n_messages == "auto":
            # Find when the agent last spoke
            num_messages_to_scan = 0
            for message in reversed(messages):
                if "role" not in message:
                    break
                elif message["role"] != "user":
                    break
                else:
                    num_messages_to_scan += 1
        num_messages_to_scan = min(len(messages), num_messages_to_scan)
        messages_to_scan = messages[-num_messages_to_scan:]

        assert len(messages_to_scan) == 1, "Only one message should be passed to the code executor."
        # iterate through the last n messages in reverse
        # if code blocks are found, execute the code blocks and return the output
        # if no code blocks are found, continue
        for message in reversed(messages_to_scan):
            if not message["content"]:
                continue
            code_blocks = self._code_executor.code_extractor.extract_code_blocks(message["content"])
            stop_on_termination = False
            if (
                    len(code_blocks) == 0 or
                    (stop_on_termination and "<FINISHED_ALL_TASKS>" in message["content"])
            ):
                if self._confidence_level == 0:
                    self._confidence_level = 1
                    return True, self.confidence_level_guidelines()
                else:
                    # force immediate termination regardless of what LLM generates
                    self._is_termination_msg = lambda x: True
                    return True, self.final_answer_guidelines()
            if self.max_turns is not None and self.turns >= self.max_turns - 1:
                # one before final allowed turn, force LLM to stop
                self._is_termination_msg = lambda x: True
                return True, self.final_answer_guidelines()

            num_code_blocks = len(code_blocks)
            if num_code_blocks == 1:
                iostream.print(
                    colored(
                        f"\n\n**EXECUTING CODE BLOCK (inferred language is {code_blocks[0].language})**\n\n",
                        "red",
                    ),
                    flush=True,
                )
            else:
                iostream.print(
                    colored(
                        f"\n\n**EXECUTING {num_code_blocks} CODE BLOCKS (inferred languages are [{', '.join([x.language for x in code_blocks])}])**\n\n",
                        "red",
                    ),
                    flush=True,
                )

            # found code blocks, execute code.
            code_result = self._code_executor.execute_code_blocks(code_blocks)
            exitcode2str = "execution succeeded" if code_result.exit_code == 0 else "execution failed"
            return True, f"exitcode: {code_result.exit_code} ({exitcode2str})\nCode output: {code_result.output}"

        return False, None

    @staticmethod
    def confidence_level_guidelines() -> str:
        return """
<confidence_guidelines>

* Give a step-by-step critique your entire response given the user's original query and any formatting constraints for constrained output.
* Consider if you used agent_tools that would have been useful, if python packages could have been used that would be useful, algorithms or code that could have been useful, etc.
* If you have a very high confidence in the response and constrained output, then say so and stop the conversation.
* However, if you do not have a very high confidence in the constrained output but do have high confidence in your response otherwise, fix the constrained output and stop the conversation.
* However, if you do not have a very high confidence in the response to the user's original query, then you must provide an executable code that would help improve your response until you have very high confidence.
* If you end up not being able to verify your response with very high confidence, but you already came up with an unverified response, give the user the unverified response (with any unverified constrained output) and provide insights and recommendations.
* For any constrained output, be sure to follow the original user query for any formatting or content constraints.
* Place a final confidence level brief summary inside <confidence> </confidence> XML tags.
* If you have already given a critique in response to these guidelines in our overall conversation, then you do not need to repeat the critique in your response.

</confidence_guidelines>

"""

    @staticmethod
    def final_answer_guidelines() -> str:
        return """
You should terminate the chat with your final answer.

<final_answer_guidelines>

* Your answer should start by answering the user's first request.
* You should give a well-structured and complete answer, insights gained, and recommendations suggested.
* Don't mention things like 'user's initial query', 'I'm sharing this again', 'final request' or 'Thank you for running the code' etc., because that wouldn't sound like you are directly talking to the user about their query.
* If no good answer was found, discuss the failures, give insights, and provide recommendations.
* If the user was asking you to write codes, make sure to provide the non-executable code block in the final answer.
* If the user was asking for images and images were made, you must add them as inline markdown using ![image](filename.png).
* If possible, use well-structured markdown as table of results or lists to make it more readable and easy to follow.
* If you have given a <constrained_output> response, please repeat that.
* You must give a very brief natural language title near the end of your response about your final answer and put that title inside <turn_title> </turn_title> XML tags.

</final_answer_guidelines>

"""


class H2OGroupChatManager(GroupChatManager):
    @backoff.on_exception(backoff.expo,
                          Exception,
                          max_tries=5,
                          giveup=lambda e: not any(re.search(pattern, str(e)) for pattern in error_patterns),
                          on_backoff=backoff_handler)
    def _generate_oai_reply_from_client(self, llm_client, messages, cache) -> typing.Union[str, typing.Dict, None]:
        try:
            return super()._generate_oai_reply_from_client(llm_client, messages, cache)
        except Exception as e:
            if any(re.search(pattern, str(e)) for pattern in error_patterns):
                logger.info(f"Encountered retryable error: {str(e)}")
                raise  # Re-raise the exception to trigger backoff
            else:
                logger.error(f"Encountered non-retryable error: {str(e)}")
                raise  # If it doesn't match our patterns, raise the original exception


def terminate_message_func(msg):
    # in conversable agent, roles are flipped relative to actual OpenAI, so can't filter by assistant
    #        isinstance(msg.get('role'), str) and
    #        msg.get('role') == 'assistant' and
    has_message = isinstance(msg, dict) and isinstance(msg.get('content', ''), str)
    has_execute = has_message and '# execution: true' in msg.get('content', '')
    if has_execute:
        # sometimes model stops without verifying results if it dumped all steps in one turn
        # force it to continue
        return False

    return False


async def get_autogen_response(func=None, use_process=False, **kwargs):
    # raise ValueError("Testing Error Handling 1")  # works

    gen_kwargs = convert_gen_kwargs(kwargs)
    kwargs = gen_kwargs.copy()
    assert func is not None, "func must be provided"
    gen = iostream_generator(func, use_process=use_process, **kwargs)

    ret_dict = {}
    async for res in gen:
        if isinstance(res, dict):
            ret_dict = res
        else:
            yield res
        await asyncio.sleep(0.005)
    yield ret_dict


def get_code_executor(
        autogen_run_code_in_docker=False,
        autogen_timeout=120,
        agent_system_site_packages=None,
        autogen_code_restrictions_level=0,
        agent_work_dir=None,
        agent_venv_dir=None,
        agent_tools_usage_hard_limits={},
        agent_tools_usage_soft_limits={},
        max_stream_length=4096,
        # max memory per code execution process
        max_memory_usage=16 * 1024 ** 3,  # 16GB
):
    if agent_work_dir is None:
        agent_work_dir = tempfile.mkdtemp()

    if autogen_run_code_in_docker:
        from autogen.coding import DockerCommandLineCodeExecutor
        # Create a Docker command line code executor.
        executor = DockerCommandLineCodeExecutor(
            image="python:3.10-slim-bullseye",
            timeout=autogen_timeout,  # Timeout for each code execution in seconds.
            work_dir=agent_work_dir,  # Use the temporary directory to store the code files.
        )
    else:
        set_python_path()
        from autogen.code_utils import create_virtual_env
        if agent_venv_dir is None:
            username = str(uuid.uuid4())
            agent_venv_dir = ".venv_%s" % username
        env_args = dict(system_site_packages=agent_system_site_packages,
                        with_pip=True,
                        symlinks=True)
        if not in_pycharm():
            virtual_env_context = create_virtual_env(agent_venv_dir, **env_args)
        else:
            print("in PyCharm, can't use virtualenv, so we use the system python", file=sys.stderr)
            virtual_env_context = None
        # work_dir = ".workdir_%s" % username
        # PythonLoader(name='code', ))

        # Create a local command line code executor.
        executor = H2OLocalCommandLineCodeExecutor(
            timeout=autogen_timeout,  # Timeout for each code execution in seconds.
            virtual_env_context=virtual_env_context,
            work_dir=agent_work_dir,  # Use the temporary directory to store the code files.
            autogen_code_restrictions_level=autogen_code_restrictions_level,
            agent_tools_usage_hard_limits=agent_tools_usage_hard_limits,
            agent_tools_usage_soft_limits=agent_tools_usage_soft_limits,
            max_stream_length=max_stream_length,
            max_memory_usage=max_memory_usage,
        )
    return executor


def merge_group_chat_messages(a, b):
    """
    Helps to merge chat messages from two different sources.
    Mostly messages from Group Chat Managers.
    """
    # Create a copy of b to avoid modifying the original list
    merged_list = b.copy()

    # Convert b into a set of contents for faster lookup
    b_contents = {item['content'] for item in b}

    # Iterate through the list a
    for i, item_a in enumerate(a):
        content_a = item_a['content']

        # If the content is not in b, insert it at the correct position
        if content_a not in b_contents:
            # Find the position in b where this content should be inserted
            # Insert right after the content of the previous item in list a (if it exists)
            if i > 0:
                prev_content = a[i - 1]['content']
                # Find the index of the previous content in the merged list
                for j, item_b in enumerate(merged_list):
                    if item_b['content'] == prev_content:
                        merged_list.insert(j + 1, item_a)
                        break
            else:
                # If it's the first item in a, just append it to the beginning
                merged_list.insert(0, item_a)

            # Update the b_contents set
            b_contents.add(content_a)

    return merged_list


def get_all_conversable_agents(group_chat_manager: GroupChatManager) -> List[ConversableAgent]:
    """
    Get all conversable agents from a group chat manager and its sub-managers.
    """
    all_conversable_agents = []
    for agent in group_chat_manager.groupchat.agents:
        if isinstance(agent, GroupChatManager):
            all_conversable_agents += get_all_conversable_agents(agent)
        else:
            all_conversable_agents.append(agent)
    return all_conversable_agents


def get_autogen_use_planning_prompt(model: str) -> bool:
    """
    Based on the model and H2OGPT_DISABLE_PLANNING_STEP environment variable, decide if autogen should use planning prompt/step.
    """
    import os
    planning_models = ['claude-3-opus', 'claude-3-5-sonnet', 'gpt-4o', 'o1-preview', 'o1-mini']
    # any pattern matching
    if any(x in model for x in planning_models):
        # sonnet35 doesn't seem to benefit
        autogen_use_planning_prompt = False
    else:
        autogen_use_planning_prompt = True if os.getenv('H2OGPT_DISABLE_PLANNING_STEP') is None else False
    return autogen_use_planning_prompt
