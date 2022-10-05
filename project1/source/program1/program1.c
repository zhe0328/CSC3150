#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){

	/* fork a child process */
	pid_t pid;
    int status;
	printf("Process start to fork\n");
	pid = fork();
	
	if (pid<0){
		printf("Fork error!\n");
		exit(1);
	}
	else {
		// child process
		if (pid==0) {
			printf("I'm the child process, my pid = %d \n", getpid());
		}
		else {
			printf("I'm the parent process, my pid = %d \n", getpid());
		}
	}
	
	/* execute test program */ 
	if (pid==0) {
        int i;
        char *arg[argc];
        
        for(i=0;i<argc-1;i++){
            arg[i]=argv[i+1];
        }
        
        arg[argc-1]=NULL;
        printf("Child process start to execute the program\n");
        execve(arg[0],arg,NULL);
        perror("execve");
	}
    else {
        /* wait for child process terminates */
        waitpid(pid, &status, WUNTRACED);
        printf("Parent process receiving the SIGCHLD signal\n");
        /* check child process'  termination status */
        if(WIFEXITED(status)){
            printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
        }
        else if (WIFSIGNALED(status)){
            if (WTERMSIG(status)==1){
                printf("child process get SIGHUP signal\n");
                printf("child process is abort by hangup signal\n");
            }
            else if (WTERMSIG(status)==2){
                printf("child process get SIGINT signal\n");
                printf("child process is abort by interrupt signal\n");
            }
            else if (WTERMSIG(status)==3){
                printf("child process get SIGQUIT signal\n");
                printf("child process is abort by quit signal\n");
            }
            else if (WTERMSIG(status)==4){
                printf("child process get SIGILL signal\n");
                printf("child process is abort by illegal instruction signal\n");
            }
            else if (WTERMSIG(status)==5){
                printf("child process get SIGTRAP signal\n");
                printf("child process is abort by trap signal\n");
            }
            else if (WTERMSIG(status)==6){
                printf("child process get SIGABRT signal\n");
                printf("child process is abort by abort signal\n");
            }
            else if (WTERMSIG(status)==7){
                printf("child process get SIGBUS signal\n");
                printf("child process is abort by bus signal\n");
            }
            else if (WTERMSIG(status)==8){
                printf("child process get SIGFPE signal\n");
                printf("child process is abort by floating signal\n");
            }
            else if (WTERMSIG(status)==9){
                printf("child process get SIGKILL signal\n");
                printf("child process is abort by kill signal\n");
            }
            else if (WTERMSIG(status)==11){
                printf("child process get SIGSEGV signal\n");
                printf("child process is abort by segment fault signal\n");
            }
            else if (WTERMSIG(status)==13){
                printf("child process get SIGPIPE signal\n");
                printf("child process is abort by pipe signal\n");
            }
            else if (WTERMSIG(status)==14){
                printf("child process get SIGALRM signal\n");
                printf("child process is abort by alarm signal\n");
            }
            else if (WTERMSIG(status)==15){
                printf("child process get SIGTERM signal\n");
                printf("child process is abort by terminate signal\n");
            }
            else {
                printf("This program does not prepare for this signal %d\n",WTERMSIG(status));
            }
            printf("CHILD EXECUTION FAILED\n");
        }
        else if (WIFSTOPPED(status)){
            if (WSTOPSIG(status)==19){
                printf("child process get SIGSTOP signal\n");
                printf("child process stopped\n");
            }
            else {
                printf("This program does not prepare for this signal %d\n",WSTOPSIG(status));
            }
            printf("CHILD PROCESS STOPPED\n");
        }
        else {
            printf("CHILD PROCESS CONTINUED\n");
        }
    }
}
