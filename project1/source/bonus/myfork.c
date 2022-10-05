#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>

int flag=0;

struct Node {
    int index;
    char filename[100];
    pid_t my_pid;
    pid_t child_pid;
    struct Node *next; // the child of the current node
};

struct statusNode { // a linked list contains process with its pid, status
    pid_t my_pid;
    pid_t parent_pid;
    char type[100]; // the type: WIFEXITED, WIFSIGNALED or WIFSTOPPED
    int code; // 0, WTERMSIG(status) or WSTOPSIG(status)
    struct statusNode * next_status;
};

void process_tree(struct statusNode * head);
int my_fork(struct Node* head, struct Node* tail, struct Node* cur, int count, int pid);

void process_tree(struct statusNode * head){
    int pid_array[100]={0};
    int index=0;
    FILE *fp = NULL;
    char buff;
    int pid;
    int ppid;
    char status[100];
    int code;
    fp = fopen("tmp.txt", "r");
    struct statusNode * current_node = (struct statusNode *) malloc(sizeof(struct statusNode));
    current_node=head;
    while(!feof(fp)){
        fscanf(fp,"%d%d%s%d",&pid,&ppid,status,&code);
        current_node->my_pid=pid;
        current_node->parent_pid=ppid;
        strcpy(current_node->type,status);
        current_node->code=code;
        current_node->next_status=(struct statusNode *) malloc(sizeof(struct statusNode));
        current_node=current_node->next_status;
        pid_array[index]=pid;
        index++;
    }
    fclose(fp);
    printf("the process tree: ");
    for (int i = index-2;i>0;i--){
        printf("%d->",pid_array[i]);
    }
    printf("%d\n",pid_array[0]);
    current_node = head;
    while (index-2>0){
        printf("The child process(pid=%d) of parent process(pid=%d) ",current_node->my_pid,current_node->parent_pid);
        if (!strcmp(current_node->type, "WIFEXITED")) {
            printf("has normal execution \nIts exit status = 0 \n \n");
        }
        else if (!strcmp(current_node->type, "WIFSIGNALED")) {
            printf("is terminated by signal\n");
            printf("Its signal number = %d\n",current_node->code);
            if (current_node->code == 1){
                printf("child process get SIGHUP signal\n");
                printf("child process is hung up\n\n");
            }
            else if (current_node->code == 2){
                printf("child process get SIGINT signal\n");
                printf("child process is interrupted\n\n");
            }
            else if (current_node->code == 3){
                printf("child process get SIGQUIT signal\n");
                printf("child process quits\n\n");
            }
            else if (current_node->code == 4){
                printf("child process get SIGILL signal\n");
                printf("child process reach an illegal instruction\n\n");
            }
            else if (current_node->code == 5){
                printf("child process get SIGTRAP signal\n");
                printf("child process reach a breakpoint\n\n");
            }
            else if (current_node->code == 6){
                printf("child process get SIGABRT signal\n");
                printf("child process is abort\n\n");
            }
            else if (current_node->code == 7){
                printf("child process get SIGBUS signal\n");
                printf("child process reach a bus\n\n");
            }
            else if (current_node->code == 8){
                printf("child process get SIGFPE signal\n");
                printf("child process is abort by floating\n\n");
            }
            else if (current_node->code == 9){
                printf("child process get SIGKILL signal\n");
                printf("child process is killed\n\n");
            }
            else if (current_node->code == 11){
                printf("child process get SIGSEGV signal\n");
                printf("child process reach a segment fault\n\n");
            }
            else if (current_node->code == 13){
                printf("child process get SIGPIPE signal\n");
                printf("child process is pipe\n\n");
            }
            else if (current_node->code == 14){
                printf("child process get SIGALRM signal\n");
                printf("child process is alarm\n\n");
            }
            else if (current_node->code == 15){
                printf("child process get SIGTERM signal\n");
                printf("child process is terminated\n\n");
            }
            else {
                printf("This program does not prepare for this signal %d\n",current_node->code);
            }
        }
        else if (!strcmp(current_node->type, "WIFSTOPPED")) {
            printf("is stopped by signal \n");
            printf("Its signal number = %d\n",current_node->code);
            if (current_node->code == 19){
                printf("child process get SIGSTOP signal\n");
                printf("child process stopped\n\n");
            }
            else {
                printf("This program does not prepare for this signal %d\n",current_node->code);
            }
        }
        current_node=current_node->next_status;
        index--;
    }
    printf("My fork process(pid=%d) execute normally\n",current_node->my_pid);
    
}

int my_fork(struct Node* head, struct Node* tail, struct Node* cur, int count, int pid)
 {
     flag += 1;
     if(flag < count) {
         pid_t pid;
         pid = fork();
         int status;
         if (pid == 0) {
             char *arg[]={NULL};
             my_fork(head, tail, cur->next, count,getpid());
             execve(cur->filename,arg,NULL);
             return 0;
         }
         else {
             waitpid(pid,&status,WUNTRACED);
             if (WIFEXITED(status)) {
                 FILE *fp = NULL;
                 char buff[255];
                 fp = fopen("tmp.txt", "a+");
                 fprintf(fp, "%d %d WIFEXITED 0\n", pid, getpid()); // my pid, my parent pid, my status my code
                 fclose(fp);
             }
             else if (WIFSIGNALED(status)){
                 FILE *fp = NULL;
                 char buff[255];
                 fp = fopen("tmp.txt", "a+");
                 fprintf(fp, "%d %d WIFSIGNALED %d\n", pid, getpid(),WTERMSIG(status));
                 fclose(fp);
             }
             else if (WIFSTOPPED(status)){
                 FILE *fp = NULL;
                 char buff[255];
                 fp = fopen("tmp.txt", "a+");
                 fprintf(fp, "%d %d WIFSTOPPED %d\n", pid, getpid(),WSTOPSIG(status));
                 fclose(fp);
             }
         }
     }
     return 0;
 }

int main(int argc,char *argv[]){
	/* Implement the functions here */
    struct Node * head = (struct Node *)malloc(sizeof(struct Node));
    struct Node * tail = (struct Node *)malloc(sizeof(struct Node));
    struct statusNode * status_node = (struct statusNode *)malloc(sizeof(struct statusNode));

    head->index = 0;
    strcpy(head->filename, argv[0]);
    tail=head;

    strcpy(status_node->type, "myfork");
    status_node->next_status=NULL;

    for (int i = 1; i < argc; i++) {
        struct Node * current_node = (struct Node *)malloc(sizeof(struct Node));
        current_node->index=i;
        strcpy(current_node->filename, argv[i]);
        current_node->next=NULL;
        tail->next=current_node;
        tail=tail->next;
    }
    tail->next = NULL;
    head->my_pid = getpid();

    struct Node * cur = (struct Node *)malloc(sizeof(struct Node));
    cur=head->next;
    my_fork(head,tail,cur,argc,getpid());
    FILE *fp = NULL;
    char buff[255];
    fp = fopen("tmp.txt", "a+");
    fprintf(fp, "%d %d MYFORK 0\n", getpid(), getppid());
    fclose(fp);
    process_tree(status_node);
	return 0;
}
