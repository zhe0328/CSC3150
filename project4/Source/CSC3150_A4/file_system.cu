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
}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	gtime++;
	// find the file in FCB
	u32 base = -1;
	int empty = -1;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
		bool find_flag = true;
		for (int j = 0; j < fs->MAX_FILENAME_SIZE; j++) {
			if (s[j] != fs->volume[base + j + 12]) {
				find_flag = false;
				base = -1;
				break;
			}
		}
		if (find_flag) {
			break;
		}
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
		fs->volume[base + 8] = 0;
		fs->volume[base + 9] = 0;
		fs->volume[base + 10] = 0;
		fs->volume[base + 11] = 0;
		// stores the file name
		for (int i = 0; i < fs->MAX_FILENAME_SIZE; i++) {
			fs->volume[base + 12 + i] = s[i];
			if (s[i] == '\0') {
				break;
			}
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
		int temp_arr[32]; // stores the a temp block of FCB
		u32 base_i, base_j, temp_base;
		int modif_time_i, modif_time_j;
		int max_mtime; // the maximum modification time
		for (int i = 0; i < fs->FCB_ENTRIES; i++) { // selection sort
			base_i = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			if (fs->volume[base_i] != '\0') {
				modif_time_i = fs->volume[base_i] + (fs->volume[base_i + 1] << 8);
				max_mtime = modif_time_i;
				for (int j = i + 1; j < fs->FCB_ENTRIES; j++) {
					base_j = fs->SUPERBLOCK_SIZE + j * fs->FCB_SIZE;
					if (fs->volume[base_j] != '\0') {
						modif_time_j = fs->volume[base_j] + (fs->volume[base_j + 1] << 8);
						if (max_mtime < modif_time_j) {
							for (int m = 0; m < fs->FCB_SIZE; m++) {
								temp_arr[m] = fs->volume[base_j + m];
							}
							temp_base = base_j;
							max_mtime = modif_time_j;
						}
					}
				}
				if (max_mtime > modif_time_i) { // exchange i-th and j-th FCB
					for (int k = 0; k < fs->FCB_SIZE; k++) {
						fs->volume[temp_base + k] = fs->volume[base_i + k];
						fs->volume[base_i + k] = temp_arr[k];
					}

				}
			}
		}
		printf("===sort by modified time===\n");
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			u32 base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			if (fs->volume[base] != '\0') {
				int j = 12;
				while (fs->volume[base + j] != '\0') {
					printf("%c", (char)fs->volume[base + j]);
					j++;
				}
				printf("\n");
			}
		}
	}
	else if (op == 1) { // LS_S
		int temp_arr[32]; // stores the a temp block of FCB
		u32 base_i, base_j;
		int fsize_i, fsize_j;
		int ctime_i, ctime_j;
		int max_fsize; // the maximum modification time
		int max_ctime;
		u32 temp_base;
		for (int i = 0; i < fs->FCB_ENTRIES; i++) { // selection sort
			base_i = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			if (fs->volume[base_i] != '\0') {
				fsize_i = fs->volume[base_i + 8] + (fs->volume[base_i + 9] << 8) + (fs->volume[base_i + 10] << 16) + (fs->volume[base_i + 11] << 24);
				ctime_i = fs->volume[base_i + 2] + (fs->volume[base_i + 3] << 8);
				max_fsize = fsize_i;
				max_ctime = ctime_i;
				for (int j = i + 1; j < fs->FCB_ENTRIES; j++) {
					base_j = fs->SUPERBLOCK_SIZE + j * fs->FCB_SIZE;
					if (fs->volume[base_j] != '\0') {
						fsize_j = fs->volume[base_j + 8] + (fs->volume[base_j + 9] << 8) + (fs->volume[base_j + 10] << 16) + (fs->volume[base_j + 11] << 24);
						ctime_j = fs->volume[base_j + 2] + (fs->volume[base_j + 3] << 8);
						if (max_fsize < fsize_j || (max_fsize == fsize_j && max_ctime > ctime_j)) {
							for (int m = 0; m < fs->FCB_SIZE; m++) {
								temp_arr[m] = fs->volume[base_j + m];
							}
							temp_base = base_j;
							max_fsize = fsize_j;
							max_ctime = ctime_j;
						}

					}
				}
				if (max_fsize > fsize_i || (max_fsize == fsize_i && max_ctime < ctime_i)) { // exchange i-th and j-th FCB
					for (int k = 0; k < fs->FCB_SIZE; k++) {
						fs->volume[temp_base + k] = fs->volume[base_i + k];
						fs->volume[base_i + k] = temp_arr[k];
					}

				}
			}
		}
		printf("===sort by file size===\n");
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			u32 base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			if (fs->volume[base] != '\0') {
				u32 file_size = fs->volume[base + 8] + (fs->volume[base + 9] << 8) + (fs->volume[base + 10] << 16) + (fs->volume[base + 11] << 24);
				int j = 12;
				while (fs->volume[base + j] != '\0') {
					printf("%c", (char)fs->volume[base + j]);
					j++;
				}
				printf(" %d\n",file_size);
			}
		}
	}
	else {
		printf("No such operation %d\n", op);
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	if (op == 2) {
		u32 base = -1;
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			bool find_flag = true;
			for (int j = 0; j < fs->MAX_FILENAME_SIZE; j++) {
				if (s[j] != fs->volume[base + j + 12]) {
					find_flag = false;
					base = -1;
					break;
				}
			}
			if (find_flag) {
				break;
			}
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
	else {
		printf("No such operation %d\n", op);
	}
}