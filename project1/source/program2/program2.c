#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

struct wait_opts {
    enum pid_type wo_type; //It is defined in ‘/include/linux/pid.h’.
    int wo_flags; //Wait options. (0, WNOHANG, WEXITED, etc.)
    struct pid *wo_pid;  //Kernel's internal notion of a process identifier. “Find_get_pid()”
    struct siginfo __user *wo_info; //Singal information.
    int __user *wo_stat; // Child process’s termination status
    struct rusage __user *wo_rusage; //Resource usage
    wait_queue_t child_wait; //Task wait queue
    int notask_error
;};

static struct task_struct *task;
extern long _do_fork(unsigned long clone_flags,
                    unsigned long stack_start,
                    unsigned long stack_size,
                    int __user *parent_tidptr,
                    int __user *child_tidptr,
                    unsigned long tls);
extern int do_execve(struct filename *filename,
                     const char __user *const __user *__argv,
                     const char __user *const __user *__envp);
extern struct filename * getname(const char __user * filename);
extern long do_wait(struct wait_opts * wo);

//implement exec function
int my_exec(void){
    /* execute a test program in child process */
    int result;
    const char path[]="/home/seed/work/assignment1/source/program2/test";
    const char *const argv[]={path,NULL,NULL};
    const char *const envp[]={"HOME=/","PATH=/sbin:/user/sbin:/bin:/usr/bin",NULL};
    struct filename * my_filename = getname(path);
    
    printk("[program2] : child process\n");
    result =do_execve(my_filename, argv, envp);
    
    // if exec success
    if (!result) return 0;
    
    // if exec failed
    do_exit(result);
    return 0;
}

//implement wait function
void my_wait(pid_t pid){
    int status;
    struct wait_opts wo;
    struct pid *wo_pid=NULL;
    enum pid_type type;
    long temp;
    
    type = PIDTYPE_PID;
    wo_pid=find_get_pid(pid);
    
    wo.wo_type=type;
    wo.wo_pid=wo_pid;
    wo.wo_flags=WEXITED|WUNTRACED;
    wo.wo_info=NULL;
    wo.wo_stat=(int __user*)&status;
    wo.wo_rusage=NULL;
    
    temp=do_wait(&wo);
    // output the child process exit status
    switch (status){
        case 4991:
            printk("[program2] : get SIGSTOP signal\n");
            printk("[program2] : child process has stop error\n");
            break;
        case 1:
            printk("[program2] : get SIGHUP signal\n");
            printk("[program2] : child process has hangup error\n");
            break;
        case 2:
            printk("[program2] : get SIGINT signal\n");
            printk("[program2] : child process has interrupt error\n");
            break;
        case 3:
            printk("[program2] : get SIGQUIT signal\n");
            printk("[program2] : child process has quit error\n");
            break;
        case 4:
            printk("[program2] : get SIGILL signal\n");
            printk("[program2] : child process has illegal instruction error\n");
            break;
        case 5:
            printk("[program2] : get SIGTRAP signal\n");
            printk("[program2] : child process has trap error\n");
            break;
        case 6:
            printk("[program2] : get SIGABRT signal\n");
            printk("[program2] : child process has abort error\n");
            break;
        case 7:
            printk("[program2] : get SIGBUS signal\n");
            printk("[program2] : child process has bus error\n");
            break;
        case 8:
            printk("[program2] : get SIGFPE signal\n");
            printk("[program2] : child process has floating error\n");
            break;
        case 9:
            printk("[program2] : get SIGKILL signal\n");
            printk("[program2] : child process has kill error\n");
            break;
        case 11:
            printk("[program2] : get SIGSEGV signal\n");
            printk("[program2] : child process has segment fault error\n");
            break;
        case 13:
            printk("[program2] : get SIGPIPE signal\n");
            printk("[program2] : child process has pipe error\n");
            break;
        case 14:
            printk("[program2] : get SIGALRM signal\n");
            printk("[program2] : child process has alarm error\n");
            break;
        case 15:
            printk("[program2] : get SIGTERM signal\n");
            printk("[program2] : child process has terminate error\n");
            break;
        default:
            printk("[program2] : This program does not prepare for this signal\n");
            break;
    }
    if (status==4991){
        printk("[program2] : The return signal is %d\n", (status&7936)>>8);
    }
    else {
        printk("[program2] : The return signal is %d\n", status);
    }
    put_pid(wo_pid);
    return;
}

//implement fork function
int my_fork(void *argc){
    pid_t pid;
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
        k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
    pid = (pid_t)_do_fork(SIGCHLD,(unsigned long)&my_exec,0,NULL,NULL,0);
    printk("[program2] : The child process has pid = %d\n",pid);
    printk("[program2] : This is the parent process, pid = %d\n",(int)current->pid);
	
	/* wait until child process terminates */
    my_wait(pid);
    
	return 0;
}

static int __init program2_init(void){

	printk("[program2] : Module_init\n");
	/* create a kernel thread to run my_fork */
    printk("[program2] : Module_init create kthread start\n");
    task=kthread_create(&my_fork,NULL,"MyFork");
    
    //wake up new thread if ok
    if(!IS_ERR(task)){
        printk("[program2] : Module_init kthread start\n");
        wake_up_process(task);
    }
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
