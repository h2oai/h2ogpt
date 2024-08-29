import re
from typing import List, Tuple

from autogen.coding import LocalCommandLineCodeExecutor


class H2OLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    @staticmethod
    def sanitize_command(lang: str, code: str) -> None:
        shell_patterns: List[Tuple[str, str]] = [
            (r"\brm\b", "Deleting files or directories is not allowed."),
            (r"\brm\s+-rf\b", "Use of 'rm -rf' command is not allowed."),
            (r"\bmv\b.*?\s+/dev/null", "Moving files to /dev/null is not allowed."),
            (r"\bdd\b", "Use of 'dd' command is not allowed."),
            (r">\s*/dev/sd[a-z][1-9]?", "Overwriting disk blocks directly is not allowed."),
            (r":\(\)\{\s*:\|\:&\s*\};:", "Fork bombs are not allowed."),
            (r"\bsudo\b", "Use of 'sudo' command is not allowed."),
            (r"\bsu\b", "Use of 'su' command is not allowed."),
            (r"\bchmod\s+([0-7]{3,4}|[+-][rwx])", "Changing file permissions is not allowed."),
            (r"\bchown\b", "Changing file ownership is not allowed."),
            (r"\bnc\b.*?\b-e\b", "Use of netcat in command execution mode is not allowed."),
            (r"\bcurl\b.*?\b\|\s*bash", "Piping curl output to bash is not allowed."),
            (r"\bwget\b.*?\b\|\s*bash", "Piping wget output to bash is not allowed."),
            (r"\b(systemctl|service)\s+(start|stop|restart)",
             "Starting, stopping, or restarting services is not allowed."),
            (r"\bnohup\b", "Use of 'nohup' command is not allowed."),
            (r"&\s*$", "Running commands in the background is not allowed."),
            (r"\bkill\b", "Use of 'kill' command is not allowed."),
            (r"\bpkill\b", "Use of 'pkill' command is not allowed."),
            (r"\b(python|python3|php|node|ruby)\s+-m\s+http\.server", "Starting an HTTP server is not allowed."),
            (r"\biptables\b", "Modifying firewall rules is not allowed."),
            (r"\bufw\b", "Modifying firewall rules is not allowed."),
            (r"\bexport\b", "Exporting environment variables is not allowed."),
            (r"\benv\b", "Accessing or modifying environment variables is not allowed."),
            (r"\becho\s+.*\s*>\s*/etc/", "Writing to system configuration files is not allowed."),
            (r"\bsed\s+.*\s+-i", "In-place file editing with sed is not allowed."),
            (r"\bawk\s+.*\s+-i", "In-place file editing with awk is not allowed."),
            (r"\bcrontab\b", "Modifying cron jobs is not allowed."),
            (r"\bat\b", "Scheduling tasks with 'at' is not allowed."),
            (r"\b(shutdown|reboot|init\s+6|telinit\s+6)\b", "System shutdown or reboot commands are not allowed."),
            (r"\b(apt-get|yum|dnf|pacman)\b", "Use of package managers is not allowed."),
            (r"\bhttp\.server\b", "Running HTTP servers is not allowed."),
        ]

        python_patterns: List[Tuple[str, str]] = [
            (r"\bos\.(remove|unlink|rmdir)\b", "Deleting files or directories is not allowed."),
            (r"\bshutil\.rmtree\b", "Deleting directory trees is not allowed."),
            (r"\bos\.system\(", "Use of os.system() is not allowed."),
            (r"\bsubprocess\.", "Use of subprocess module is not allowed."),
            (r"\bexec\(", "Use of exec() is not allowed."),
            (r"\beval\(", "Use of eval() is not allowed."),
            (r"\b__import__\(", "Use of __import__() is not allowed."),
            (r"\bopen\(.*?,\s*['\"](w|a|r\+|w\+|a\+)", "Writing to files is not allowed."),
            (r"\bos\.remove\(", "Removing files is not allowed."),
            (r"\bos\.unlink\(", "Removing files is not allowed."),
            (r"\bshutil\.rmtree\(", "Removing directory trees is not allowed."),
            (r"\bos\.rmdir\(", "Removing directories is not allowed."),
            (r"\bsocket\.", "Use of socket module is not allowed."),
            (r"\burllib\.request\.", "Use of urllib.request is not allowed."),
            (r"\brequests\.", "Use of requests library is not allowed."),
            (r"\bsmtplib\.", "Use of smtplib (for sending emails) is not allowed."),
            (r"\bctypes\.", "Use of ctypes module is not allowed."),
            (r"\bpty\.", "Use of pty module is not allowed."),
            (r"\bplatform\.", "Use of platform module is not allowed."),
            (r"\bsys\.exit\(", "Use of sys.exit() is not allowed."),
            (r"\bos\.chmod\(", "Changing file permissions is not allowed."),
            (r"\bos\.chown\(", "Changing file ownership is not allowed."),
            (r"\bos\.setuid\(", "Changing process UID is not allowed."),
            (r"\bos\.setgid\(", "Changing process GID is not allowed."),
            (r"\bos\.fork\(", "Forking processes is not allowed."),
            #(r"\bmultiprocessing\.", "Use of multiprocessing module is not allowed."),
            (r"\bsched\.", "Use of sched module (for scheduling) is not allowed."),
            (r"\bcommands\.", "Use of commands module is not allowed."),
            (r"\bpdb\.", "Use of pdb (debugger) is not allowed."),
            #(r"\bsqlite3\.", "Use of sqlite3 module is not allowed."),
            (r"\bpickle\.loads\(", "Use of pickle.loads() is not allowed."),
            (r"\bmarshall\.loads\(", "Use of marshall.loads() is not allowed."),
            #(r"\bos\.environ\b", "Accessing environment variables is not allowed."),
            (r"\bhttp\.server\b", "Running HTTP servers is not allowed."),
        ]

        if lang in ["bash", "shell", "sh"]:
            patterns = shell_patterns
        elif lang == "python":
            patterns = python_patterns
        else:
            return super().sanitize_command(lang, code)

        for pattern, message in patterns:
            if re.search(pattern, code):
                raise ValueError(f"Potentially dangerous operation detected: {message}")

        return super().sanitize_command(lang, code)
