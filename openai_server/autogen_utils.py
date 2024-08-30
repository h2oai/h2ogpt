import logging
import os
import re
import sys
import typing
from typing import List, Tuple

from autogen.coding import LocalCommandLineCodeExecutor, CodeBlock
from autogen.coding.base import CommandLineCodeResult

verbose = os.getenv('VERBOSE', '0').lower() == '1'

danger_mark = 'Potentially dangerous operation detected'
bad_output_mark = 'Output contains sensitive information'


class H2OLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
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

    def _execute_code_dont_check_setup(self, code_blocks: List[CodeBlock]) -> CommandLineCodeResult:
        try:
            ret = super()._execute_code_dont_check_setup(code_blocks)
        except Exception as e:
            if danger_mark in str(e):
                print(f"Code Danger Error: {e}\n\n{code_blocks}", file=sys.stderr)
                # dont' fail, just return the error so LLM can adjust
                return CommandLineCodeResult(exit_code=1, output=str(e))
            else:
                raise
        try:
            ret = self.output_guardrail(ret)
        except Exception as e:
            if bad_output_mark in str(e):
                print(f"Code Output Danger Error: {e}\n\n{code_blocks}\n\n{ret}", file=sys.stderr)
                # dont' fail, just return the error so LLM can adjust
                return CommandLineCodeResult(exit_code=1, output=str(e))
            else:
                raise
        ret = self.truncate_output(ret)
        return ret

    @staticmethod
    def output_guardrail(ret: CommandLineCodeResult) -> CommandLineCodeResult:
        # List of API key environment variable names to check
        api_key_names = ['OPENAI_AZURE_KEY', 'TWILIO_AUTH_TOKEN', 'NEWS_API_KEY', 'OPENAI_API_KEY_JON',
                         'H2OGPT_H2OGPT_KEY', 'TWITTER_API_KEY', 'FACEBOOK_ACCESS_TOKEN', 'API_KEY', 'LINKEDIN_API_KEY',
                         'STRIPE_API_KEY', 'ADMIN_PASS', 'S2_API_KEY', 'ANTHROPIC_API_KEY', 'AUTH_TOKEN',
                         'AWS_SERVER_PUBLIC_KEY', 'OPENAI_API_KEY', 'HUGGING_FACE_HUB_TOKEN', 'AWS_ACCESS_KEY_ID',
                         'SERPAPI_API_KEY', 'WOLFRAM_ALPHA_APPID', 'AWS_SECRET_ACCESS_KEY', 'ACCESS_TOKEN',
                         'SLACK_API_TOKEN', 'MISTRAL_API_KEY', 'TOGETHERAI_API_TOKEN', 'GITHUB_TOKEN', 'SECRET_KEY',
                         'GOOGLE_API_KEY', 'REPLICATE_API_TOKEN', 'GOOGLE_CLIENT_SECRET', 'GROQ_API_KEY',
                         'AWS_SERVER_SECRET_KEY', 'H2OGPT_OPENAI_BASE_URL', 'H2OGPT_OPENAI_API_KEY',
                         'H2OGPT_OPENAI_PORT', 'H2OGPT_OPENAI_HOST', 'H2OGPT_OPENAI_CERT_PATH',
                         'H2OGPT_OPENAI_KEY_PATH', 'H2OGPT_MAIN_KWARGS',
                         'GRADIO_H2OGPT_H2OGPT_KEY']

        # Get the values of these environment variables
        set_api_key_names = set(api_key_names)
        set_api_key_values = set([os.getenv(key, '') for key in set_api_key_names])

        # Expanded set of allowed (dummy) values
        set_allowed = {
            '', 'EMPTY', 'DUMMY', None, 'null', 'NULL', 'Null',
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
            'undefined', 'UNDEFINED',
            # Add any other common dummy values you've encountered
        }

        # Filter out allowed (dummy) values
        api_key_values = [value for value in set_api_key_values if value and value not in set_allowed]

        if ret.output:
            # Check if any API key value is in the output and collect all violations
            violated_keys = []
            for api_key in api_key_values:
                if api_key in ret.output:
                    # Find the corresponding key name(s) for the violated value
                    violated_key_names = [name for name in api_key_names if os.getenv(name) == api_key]
                    violated_keys.extend(violated_key_names)

            # If any violations were found, raise an error with all violated keys
            if violated_keys:
                error_message = f"Output contains sensitive information. Violated keys: {', '.join(violated_keys)}"
                raise ValueError(error_message)

        return ret

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


# override the original method with the new one
# required because original class does not use super() but references its own method
LocalCommandLineCodeExecutor.sanitize_command = H2OLocalCommandLineCodeExecutor.sanitize_command

from autogen import ConversableAgent
import backoff

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
