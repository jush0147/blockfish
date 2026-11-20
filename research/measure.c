/* Per-race CPU/RSS accounting without a GNU time dependency. The benchmark
 * starts a new process group and kills that whole group on timeout. */
#define _DEFAULT_SOURCE
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

int main(int argc, char **argv) {
    if (argc < 3) return 125;
    pid_t child = fork();
    if (child < 0) return 125;
    if (child == 0) { execvp(argv[2], argv+2); perror("execvp"); _exit(127); }
    int status;
    struct rusage usage;
    while (wait4(child, &status, 0, &usage) < 0) {
        if (errno != EINTR) return 125;
    }
    FILE *out = fopen(argv[1], "w");
    if (!out) return 125;
    fprintf(out, "{\"user_seconds\":%.6f,\"system_seconds\":%.6f,\"max_rss_kib\":%ld}\n",
        usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1e6,
        usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1e6, usage.ru_maxrss);
    fclose(out);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return WIFSIGNALED(status) ? 128+WTERMSIG(status) : 125;
}
