#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE; // 4096 volumes
  fs->FCB_SIZE = FCB_SIZE; // name: 20, size: 4, starting address: 4, create time: 2, modification time: 2
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  fs->FILE_STARTING_ADDRESS = FILE_BASE_ADDRESS;
  fs->MAX_DEPTH = 3;
  fs->root = new struct dir;
  fs->root->child = NULL;
  fs->root->parent = NULL;
  fs->root->sibling = NULL;
  fs->root->index = 0;
  fs->root->flag = 0;
  fs->root->base = -1;
  fs->root->size = 0;
  fs->cur = fs->root;
}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	gtime++;
	// find the file in FCB
	u32 base = -1;
	dir * cur_child = fs->cur->child;
	while (cur_child != NULL) { // search the file in the current directory
		bool find_flag = true;
		for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
			if (s[i] != cur_child->name[i]) {
				find_flag = false;
				break;
			}
		}
		if (find_flag) {
			base = cur_child->base;
			break;
		}
		cur_child = cur_child->sibling;
	}
	if (base != -1) { // the file exists
		// update modification time
		fs->volume[base] = gtime % 256;
		fs->volume[base + 1] = (gtime >> 8) % 256;
		u32 addr = fs->volume[base + 4] + (fs->volume[base + 5] << 8)+ (fs->volume[base + 6] << 16) + (fs->volume[base + 7] << 24);
		return addr;
	}
	else if (op == 0){
		printf("The open file %s is not found!\n",s);
		return -1;
	}
	else if (op == 1){ // the file does not exist and op is write
		u32 start_addr = fs->FILE_STARTING_ADDRESS; // the starting physical address of the file
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			if (fs->volume[base] == '\0') { // find an empty FCB
				break;
			}
		}
		u32 size = 0;
		// stores the modification time
		fs->volume[base] = gtime % 256;
		fs->volume[base + 1]= (gtime >> 8) % 256;
		// stores the create time
		fs->volume[base + 2] = gtime % 256;
		fs->volume[base + 3] = (gtime >> 8) % 256;
		// stores the starting physical address
		fs->volume[base + 4] = start_addr % 256;
		fs->volume[base + 5] = (start_addr >> 8) % 256;
		fs->volume[base + 6] = (start_addr >> 16) % 256;
		fs->volume[base + 7] = (start_addr >> 24) % 256;
		// stores the file size (0)
		fs->volume[base + 8] = fs->volume[base + 9] = fs->volume[base + 10] = fs->volume[base + 11] = 0;
		// stores the file name
		dir *new_file = new struct dir;
		int name_size = 0;
		for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
			fs->volume[base + 12 + i] = s[i];
			new_file->name[i] = s[i];
			name_size++;
			if (s[i] == '\0') {
				break;
			}
		}
		new_file->flag = 1;
		new_file->index = fs->cur->index + 1;
		new_file->size = 0;
		new_file->base = base;
		new_file->parent = fs->cur;
		new_file->child = NULL;
		new_file->sibling = NULL;
		dir *cur_child = fs->cur->child;
		if (cur_child == NULL) {
			fs->cur->child = new_file;
		}
		else {
			while (cur_child->sibling != NULL) {
				cur_child = cur_child->sibling;
			}
			cur_child->sibling = new_file;
		}
		dir * cur_parent = fs->cur;
		cur_parent->size += name_size;
		// update the size of its parent (cur) directory in FCB
		if (fs->cur != fs->root) {
			u32 dir_base = fs->cur->base;
			u32 dir_size = fs->volume[dir_base + 8] + (fs->volume[dir_base + 9] << 8) + (fs->volume[dir_base + 10] << 16) + (fs->volume[dir_base + 11] << 24);
			dir_size += name_size;
			fs->volume[dir_base + 8] = dir_size % 256;
			fs->volume[dir_base + 9] = (dir_size >> 8) % 256;
			fs->volume[dir_base + 10] = (dir_size >> 16) % 256;
			fs->volume[dir_base + 11] = (dir_size >> 24) % 256;
		}
		return start_addr;
	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[fp++];
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp) // fp: physical starting address
{
	/* Implement write operation here */
	gtime++;
	u32 base = -1;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) { // to find the base
		base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
		if (fs->volume[base] != '\0') { // the FCB entry is not empty
			u32 addr = fs->volume[base + 4] + (fs->volume[base + 5] << 8) + (fs->volume[base + 6] << 16) + (fs->volume[base + 7] << 24);
			if (addr == fp) {
				break;
			}
		}
	}
	// update the modification time
	fs->volume[base] = gtime % 256;
	fs->volume[base + 1] = (gtime >> 8) % 256;
	// get the size of the old file
	u32 old_file_size = fs->volume[base + 8] + (fs->volume[base + 9] << 8) + (fs->volume[base + 10] << 16) + (fs->volume[base + 11] << 24);
	// update the new size of the file
	fs->volume[base + 8] = size % 256;
	fs->volume[base + 9] = (size >> 8) % 256;
	fs->volume[base + 10] = (size >> 16) % 256;
	fs->volume[base + 11] = (size >> 24) % 256;

	if (fs->FILE_STARTING_ADDRESS == fs->FILE_BASE_ADDRESS) { // the content of the files is empty
		// write the disk
		for (int i = 0; i < size; i++) {
			fs->volume[fp++] = input[i];
		}
		fs->FILE_STARTING_ADDRESS += size;
	}
	else if (fs->FILE_STARTING_ADDRESS == fp + old_file_size) { // the writing file is at the end of the volume
		// write the disk
		for (int i = 0; i < size; i++) {
			fs->volume[fp+i] = input[i];
		}
		if (size < old_file_size) {
			for (int i = fp + old_file_size - size; i < fp + old_file_size; i++) {
				fs->volume[i] = '\0';
			}
		}
		fs->FILE_STARTING_ADDRESS += (size-old_file_size);
	}
	else {
		for (u32 i = fp + old_file_size; i < fs->FILE_STARTING_ADDRESS; i++) { // compaction of contents for other files
			fs->volume[i - old_file_size] = fs->volume[i];
		}
		for (int j = 0; j < size; j++) { //rewrite the content of the files and put it in the end
			fs->volume[j + fs->FILE_STARTING_ADDRESS-old_file_size] = input[j];
		}
		if (size < old_file_size) {
			for (int i = 0; i < old_file_size - size; i++) {
				fs->volume[fs->FILE_STARTING_ADDRESS - i] = '\0';
			}
		}
		// in FCB, update the starting physical address
		u32 new_write_addr= fs->FILE_STARTING_ADDRESS - old_file_size; // the new physical address of the rewritten file
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			u32 other_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			u32 addr = fs->volume[other_base + 4] + (fs->volume[other_base + 5] << 8) + (fs->volume[other_base + 6] << 16) + (fs->volume[other_base + 7] << 24);
			if (fs->volume[other_base] != '\0' && base != other_base && addr >= fp + old_file_size) { // update other FCB's physical address
				addr -= old_file_size;
				fs->volume[other_base + 4] = addr % 256;
				fs->volume[other_base + 5] = (addr >> 8) % 256;
				fs->volume[other_base + 6] = (addr >> 16) % 256;
				fs->volume[other_base + 7] = (addr >> 24) % 256;
			}
		}
		// update the starting physical address of the rewritten file
		fs->volume[base + 4] = new_write_addr % 256;
		fs->volume[base + 5] = (new_write_addr >> 8) % 256;
		fs->volume[base + 6] = (new_write_addr >> 16) % 256;
		fs->volume[base + 7] = (new_write_addr >> 24) % 256;

		fs->FILE_STARTING_ADDRESS += (size - old_file_size);	
	}
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	if (!op) { // LS_D
		int max_mtime; // the maximum modification time
		if (fs->cur->child == NULL) {
			printf("===sort by modified time===\n");
			return;
		}
		dir * temp_sibling = fs->cur->child;
		int temp_base[1024];
		int temp_mtime[1024];
		int cur=0; // current size
		int max_index;
		while (temp_sibling != NULL) {
			int base = temp_sibling->base;
			int mtime = fs->volume[base] + (fs->volume[base + 1] << 8);
			temp_mtime[cur] = mtime;
			temp_base[cur] = base;
			temp_sibling = temp_sibling->sibling;
			cur++;
		}
		//printf("cur is %d\n", cur);
		for (int i = 0; i < cur; i++) {
			max_mtime = temp_mtime[i];
			max_index = i;
			for (int j = i + 1; j < cur; j++) {
				if (temp_mtime[j] > max_mtime) {
					max_mtime = temp_mtime[j];
					max_index = j;
				}
			}
			if (max_index != i) { //exchange i-th and j-th
				int temp_j = temp_base[max_index];
				temp_mtime[max_index] = temp_mtime[i];
				temp_base[max_index] = temp_base[i];
				temp_mtime[i] = max_mtime;
				temp_base[i] = temp_j;
			}
		}
		printf("===sort by modified time===\n");
		for (int k = 0; k < cur; k++) {
			int base = temp_base[k];
			bool dir_flag = true;
			if (fs->volume[base] != '\0') {
				int j = 12;
				while (fs->volume[base + j] != '\0') {
					if (fs->volume[base + j] == 46) {
						dir_flag = false;
					}
					printf("%c", (char)fs->volume[base + j]);
					j++;
				}
				if (dir_flag) {
					printf(" d");
				}
				printf("\n");
			}
		}
	}
	else if (op == 1) { // LS_S
		if (fs->cur->child == NULL) {
			printf("===sort by file size===\n");
			return;
		}
		dir * temp_sibling = fs->cur->child;
		int temp_base[1024];
		int temp_fsize[1024];
		int temp_ctime[1024];
		int cur = 0; // current size
		int max_index;
		int max_fsize;
		int max_ctime;
		while (temp_sibling != NULL) {
			int base = temp_sibling->base;
			int fsize;
			if (temp_sibling->flag == 1) { // file
				fsize = fs->volume[base + 8] + (fs->volume[base + 9] << 8) + (fs->volume[base + 10] << 16) + (fs->volume[base + 11] << 24);
			}
			else { // directory
				fsize = temp_sibling->size;
			}
			int ctime = fs->volume[base + 2] + (fs->volume[base + 3] << 8);
			temp_fsize[cur] = fsize;
			temp_base[cur] = base;
			temp_ctime[cur] = ctime;
			temp_sibling = temp_sibling->sibling;
			cur++;
		}
		for (int i = 0; i < cur; i++) {
			max_fsize = temp_fsize[i];
			max_ctime = temp_ctime[i];
			max_index = i;
			for (int j = i + 1; j < cur; j++) {
				if (temp_fsize[j] > max_fsize || (max_fsize == temp_fsize[j] && max_ctime > temp_ctime[j])) {
					max_fsize = temp_fsize[j];
					max_ctime = temp_ctime[j];
					max_index = j;
				}
			}
			if (max_index != i) { //exchange i-th and j-th
				int temp_j = temp_base[max_index];
				temp_fsize[max_index] = temp_fsize[i];
				temp_ctime[max_index] = temp_ctime[i];
				temp_base[max_index] = temp_base[i];
				temp_fsize[i] = max_fsize;
				temp_ctime[i] = max_ctime;
				temp_base[i] = temp_j;
			}
		}
		printf("===sort by file size===\n");
		for (int k = 0; k < cur; k++) {
			int base = temp_base[k];
			int file_size = temp_fsize[k];
			bool dir_flag = true;
			if (fs->volume[base] != '\0') {
				int j = 12;
				while (fs->volume[base + j] != '\0') {
					if (fs->volume[base + j] == 46) {
						dir_flag = false;
					}
					printf("%c", (char)fs->volume[base + j]);
					j++;
				}
				if (dir_flag) {
					printf(" %d d\n", file_size);
				}
				else {
					printf(" %d\n", file_size);
				}
			}
		}
	}
	else if (op == 5) { // move up to the parent directory
		fs->cur = fs->cur->parent;
	}
	else if (op == 7) {
		int temp_name[1024];
		int temp_size[4];
		int cur = 0;
		int index = 0;
		dir* temp = fs->cur;
		while (temp != NULL && temp !=fs->root) {
			for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
				if (temp->name[i] == '\0') {
					temp_name[cur] = 47;
					temp_size[index] = cur;
					cur++;
					break;
				}
				temp_name[cur] = temp->name[i];
				cur++;
			}
			index++;
			temp = temp->parent;
		}
		for (int j = index-1; j > 0; j--) {
			for (int k = temp_size[j-1]; k < temp_size[j]; k++) {
				printf("%c", (char)temp_name[k]);
			}
		}
		printf("/");
		for (int k = 0; k < temp_size[0]; k++) {
			printf("%c", (char)temp_name[k]);
		}
		printf("\n");
	}
	else {
		printf("No such operation %d\n", op);
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	if (op == 2) {
		/* Implement rm operation here */
		u32 base = -1;
		dir * cur_child = fs->cur->child;
		while (cur_child != NULL) { // search the file in the current directory
			bool find_flag = true;
			for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
				if (s[i] != cur_child->name[i]) {
					find_flag = false;
					break;
				}
			}
			if (find_flag && cur_child->flag == 1) {
				base = cur_child->base;
				break;
			}
			cur_child = cur_child->sibling;
		}
		if (base == -1) {
			printf("File %s does not exist!\n", s);
			return;
		}
		else {
			// release the file space
			u32 addr = fs->volume[base + 4] + (fs->volume[base + 5] << 8) + (fs->volume[base + 6] << 16) + (fs->volume[base + 7] << 24);
			u32 file_size = fs->volume[base + 8] + (fs->volume[base + 9] << 8) + (fs->volume[base + 10] << 16) + (fs->volume[base + 11] << 24);
			for (u32 i = addr + file_size; i < fs->FILE_STARTING_ADDRESS; i++) { // compaction of contents for other files
				fs->volume[i - file_size] = fs->volume[i];
			}
			for (u32 i = fs->FILE_STARTING_ADDRESS - file_size; i < fs->FILE_STARTING_ADDRESS; i++) {
				fs->volume[i] = '\0';
			}
			// in FCB, update the starting physical address
			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				u32 other_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
				u32 other_addr = fs->volume[other_base + 4] + (fs->volume[other_base + 5] << 8) + (fs->volume[other_base + 6] << 16) + (fs->volume[other_base + 7] << 24);
				if (fs->volume[other_base] != '\0' && base != other_base && other_addr >= addr + file_size) { // update other FCB's physical address
					other_addr -= file_size;
					fs->volume[other_base + 4] = other_addr % 256;
					fs->volume[other_base + 5] = (other_addr >> 8) % 256;
					fs->volume[other_base + 6] = (other_addr >> 16) % 256;
					fs->volume[other_base + 7] = (other_addr >> 24) % 256;
				}
			}
			//release the FCB
			for (int i = 0; i < fs->FCB_SIZE; i++) {
				fs->volume[base + i] = '\0';
			}
			fs->FILE_STARTING_ADDRESS -= file_size;
		}
	}
	else if (op == 3) { // create a directory
		if (fs->cur->index < fs->MAX_DEPTH) {
			gtime++;
			u32 base;
			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
				if (fs->volume[base] == '\0') { // find an empty FCB
					break;
				}
			}
			// stores the modification time
			fs->volume[base] = gtime % 256;
			fs->volume[base + 1] = (gtime >> 8) % 256;
			// stores the create time
			fs->volume[base + 2] = gtime % 256;
			fs->volume[base + 3] = (gtime >> 8) % 256;
			// stores the starting physical address
			fs->volume[base + 4] = fs->volume[base + 5] = fs->volume[base + 6] = fs->volume[base + 7] = -1;
			// stores the dir size (0)
			fs->volume[base + 8] = fs->volume[base + 9] = fs->volume[base + 10] = fs->volume[base + 11] = 0;
			// stores the dir name
			int dir_size = 0;
			dir *new_dir = new struct dir;
			for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
				fs->volume[base + 12 + i] = s[i];
				new_dir->name[i] = s[i];
				dir_size++;
				if (s[i] == '\0') {
					break;
				}
			}
			new_dir->size = 0;
			new_dir->flag = 0;
			new_dir->base = base;
			new_dir->index = fs->cur->index + 1;
			new_dir->parent = fs->cur;
			struct dir *cur_child = fs->cur->child;
			if (cur_child == NULL) {
				fs->cur->child = new_dir;
			}
			else {
				while (cur_child->sibling != NULL) {
					cur_child = cur_child->sibling;
				}
				cur_child->sibling = new_dir;
			}
			new_dir->sibling = NULL;
			new_dir->child = NULL;;
			fs->cur->size += dir_size;
		}
		else {
			printf("The file system reaches its maximum depth %d\n", fs->MAX_DEPTH);
		}
	}
	else if (op == 4) { // enter a subdirectory with name s
		dir *cur_child = fs->cur->child;
		while (cur_child != NULL) { //find the directory
			bool find_flag = true;
			for (int j = 0; j < fs->MAX_FILENAME_SIZE; j++) {
				if (s[j] != cur_child->name[j]) {
					find_flag = false;
					break;
				}
			}
			if (find_flag) {
				break;
			}
			cur_child = cur_child->sibling;
		}
		fs->cur = cur_child;
	}
	else if (op == 6) { // remove the directory with name s and all its subdirectories and files
		dir *cur = fs->cur;
		dir *cur_child = fs->cur->child;
		dir *left = cur->child;
		int dir_name_size = 0;
		while (left->sibling != NULL) { //find the directory
			bool find_flag = true;
			dir_name_size = 0;
			for (int j = 0; j < fs->MAX_FILENAME_SIZE; j++) {
				if (s[j] != '\0') {
					dir_name_size++;
				}
				if (s[j] != left->sibling->name[j]) {
					find_flag = false;
					break;
				}
			}
			if (find_flag) {
				break;
			}
			left = left->sibling;
		}
		dir_name_size++;
		cur_child = left->sibling;
		if (left->sibling == NULL) {
			cur_child = left;
		}
		// update the parent directory size
		dir * parent_dir = cur_child->parent;
		parent_dir->size -= dir_name_size;
		dir * sibling = cur_child->child;
		while (sibling != NULL) {
			if (sibling->flag == 0) { //a directory
				dir * new_child = sibling->child;
				while (new_child != NULL) {
					int base = new_child->base;
					int starting_addr = fs->volume[base + 4] + (fs->volume[base + 5] << 8) + (fs->volume[base + 6] << 16) + (fs->volume[base + 7] << 24);
					int file_size = fs->volume[base + 8] + (fs->volume[base + 9] << 8) + (fs->volume[base + 10] << 16) + (fs->volume[base + 11] << 24);
					// free the content and do the compaction of remaining files
					for (u32 i = starting_addr + file_size; i < fs->FILE_STARTING_ADDRESS; i++) { // compaction of contents for other files
						fs->volume[i - file_size] = fs->volume[i];
					}
					for (int i = 0; i < file_size; i++) {
						fs->volume[fs->FILE_STARTING_ADDRESS - i] = '\0';
					}
					// in FCB, update the starting physical address
					for (int i = 0; i < fs->FCB_ENTRIES; i++) {
						u32 other_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
						u32 addr = fs->volume[other_base + 4] + (fs->volume[other_base + 5] << 8) + (fs->volume[other_base + 6] << 16) + (fs->volume[other_base + 7] << 24);
						if (fs->volume[other_base] != '\0' && base != other_base && addr >= starting_addr + file_size) { // update other FCB's physical address
							addr -= file_size;
							fs->volume[other_base + 4] = addr % 256;
							fs->volume[other_base + 5] = (addr >> 8) % 256;
							fs->volume[other_base + 6] = (addr >> 16) % 256;
							fs->volume[other_base + 7] = (addr >> 24) % 256;
						}
					}

					fs->FILE_STARTING_ADDRESS -= file_size;
					// free the FCB
					for (int k = 0; k < fs->FCB_ENTRIES; k++) {
						fs->volume[k + base] = '\0';
					}
					new_child = new_child->sibling;
				}
				/*fs->cur = sibling;
				printf("sibling name %s\n", sibling->name);
				char * temp_name;
				for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
					temp_name[i] = sibling->name[i];
					if (sibling->name[i] == '\0') {
						break;
					}
				}
				// fs_gsys(fs, 6, temp_name);
				fs->cur = cur;*/
			}
			else { // a file		
				int base = sibling->base;
				int starting_addr = fs->volume[base + 4] + (fs->volume[base + 5] << 8) + (fs->volume[base + 6] << 16) + (fs->volume[base + 7] << 24);
				int file_size = fs->volume[base + 8] + (fs->volume[base + 9] << 8) + (fs->volume[base + 10] << 16) + (fs->volume[base + 11] << 24);
				// free the content and do the compaction of remaining files
				for (u32 i = starting_addr + file_size; i < fs->FILE_STARTING_ADDRESS; i++) { // compaction of contents for other files
					fs->volume[i - file_size] = fs->volume[i];
				}
				for (int i = 0; i < file_size; i++) {
					fs->volume[fs->FILE_STARTING_ADDRESS - i] = '\0';
				}
				// in FCB, update the starting physical address
				for (int i = 0; i < fs->FCB_ENTRIES; i++) {
					u32 other_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
					u32 addr = fs->volume[other_base + 4] + (fs->volume[other_base + 5] << 8) + (fs->volume[other_base + 6] << 16) + (fs->volume[other_base + 7] << 24);
					if (fs->volume[other_base] != '\0' && base != other_base && addr >= starting_addr + file_size) { // update other FCB's physical address
						addr -= file_size;
						fs->volume[other_base + 4] = addr % 256;
						fs->volume[other_base + 5] = (addr >> 8) % 256;
						fs->volume[other_base + 6] = (addr >> 16) % 256;
						fs->volume[other_base + 7] = (addr >> 24) % 256;
					}
				}

				fs->FILE_STARTING_ADDRESS -= file_size;
				// free the FCB
				for (int k = 0; k < fs->FCB_ENTRIES; k++) {
					fs->volume[k + base] = '\0';
				}
			}
			dir * free_sibling = sibling;
			sibling = free_sibling->sibling;
			delete(free_sibling);
			//fs->cur->child = sibling;
		}
		if (fs->cur->child == cur_child) { // only one directory in the current directory
			fs->cur->child = NULL;
		}
		else if (fs->cur->child == cur_child && cur_child->sibling != NULL) {
			fs->cur->child = cur_child->sibling;
			delete(cur_child);
		}
		else if (fs->cur->child->sibling == cur_child && cur_child->sibling == NULL) {
			fs->cur->child->sibling = NULL;
			delete(cur_child);
		}
		else {
			dir * child_sibling = cur_child->sibling;
			left->sibling = child_sibling;
			delete(cur_child);
		}
	}
	else {
		printf("No such operation %d\n", op);
	}
}
