#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50

pthread_mutex_t mutex;

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN] ;
int state = 0; // 0: the game is in progress, 1: win, 2: lose, 3: quit

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void *logs_move( void *t ){
    long tid;
    tid = (long) t+1;
    
	/*  Move the logs  */
    int start_column = rand()% COLUMN;
    int log_size = rand() % (COLUMN/4)+8;
    pthread_mutex_lock(&mutex);
    while (state == 0){
        int on_log = 0; // 1 represents the frog is on the log, 0 represents the frog is not on the log
        int speed_flag = rand()%2;
        if (speed_flag == 0){
            usleep(200000);
        }
        else {
            usleep(100000);
        }
        
        /*  Check keyboard hits, to change frog's position or quit the game. */
        if (kbhit()){
            char dir = getchar();
            if (frog.x == ROW && frog.y){
                map[frog.x][frog.y] = '|';
            }
            if (dir == 'w' || dir == 'W'){
                frog.x -= 1;
            }
            else if (dir == 's' || dir == 'S'){
                frog.x += 1;
            }
            else if (dir == 'a' || dir == 'A'){
                frog.y -= 1;
            }
            else if (dir == 'd' || dir == 'D'){
                frog.y += 1;
            }
            else if (dir == 'q' || dir == 'Q'){
                state = 3;
            }
        }
        
        if (tid%2==1){ // left
            if (frog.x==tid){
                frog.y--;
            }
            for (int j=log_size+start_column;j>=start_column;j--){
                if (frog.x==tid && (frog.y+1)==(j+COLUMN)%(COLUMN-1)){
                    on_log=1;
                }
                map[tid][(j+COLUMN)%(COLUMN-1)]='=';
            }
            if (frog.y+1==(start_column+COLUMN)%(COLUMN-1)){
                on_log=0;
            }
            map[tid][(log_size+start_column+COLUMN)%(COLUMN-1)]=' ';
            start_column=(start_column+COLUMN-1)%(COLUMN);
        }
        else if (tid%2==0){ // right
            if (frog.x==tid){
                frog.y++;
            }
            for (int j=start_column;j<=log_size+start_column;j++){
                if (frog.x==tid && (frog.y-1)==j%(COLUMN-1)){
                    on_log=1;
                }
                map[tid][j%(COLUMN-1)]='=';
            }
            if (frog.y==(log_size+start_column+COLUMN)%(COLUMN-1)){
                on_log=0;
            }
            map[tid][start_column%(COLUMN-1)]=' ';
            start_column=(start_column+1)%(COLUMN-1);
        }
        map[frog.x][frog.y] = '0';
        /*  Check game's status  */
        if (frog.x > ROW || frog.y < 1 || frog.y > COLUMN-2 || (!on_log && frog.x==tid)){
            state = 2;
        }
        if (frog.x == 0){
            state = 1;
        }
        
        printf("\033[H\033[2J");
        for(int i = 0; i <= ROW; i++){
            puts(map[i]);
        }
        /*  Print the map on the screen  */
        if (state==1){
            printf("\033[H\033[2JYou win the game!!\n");
        }
        else if (state==2){
            printf("\033[H\033[2JYou lose the game!!\n");
        }
        else if (state==3){
            printf("\033[H\033[2JYou exit the game.\n");
        }
        pthread_mutex_unlock(&mutex);
    }

    pthread_exit(NULL);
}

int main( int argc, char *argv[] ){

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  // 9 rows
	}
    for( j = 0; j < COLUMN - 1; ++j ){
        map[ROW][j] = map[0][j] = '|' ; // 49 columns
    }
	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ;
	map[frog.x][frog.y] = '0' ;

	//Print the map into screen
    printf("\033[H\033[2J");
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );

	/*  Create pthreads for wood move and frog control.  */
    unsigned int seed;
    seed = static_cast<unsigned int>(time(NULL));
    srand(seed);
    
    pthread_t threads[ROW-1];
    int rc;
    pthread_mutex_init(&mutex, NULL);
    for (int k=0;k<ROW-1;k++){
        rc = pthread_create(&threads[k],NULL,logs_move, (void*)k);
        if (rc){
            printf("ERROR: return code from pthread_create() is %d", rc);
            exit(1);
        }
    }
    
    pthread_mutex_destroy(&mutex);
    pthread_exit(NULL);

	return 0;

}
