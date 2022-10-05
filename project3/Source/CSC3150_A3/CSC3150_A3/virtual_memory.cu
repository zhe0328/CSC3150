#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"


__device__ void init_invert_page_table(VirtualMemory *vm) {
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0; // unused: 0 used: 1
	}
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  vm->head = NULL;
  vm->tail = NULL;
  vm->count = 0;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
	// page offset length 5, page number 27
	int offset = addr % 32;
	int base = addr >> 5;
	int target = -1;
	pageNode * cur = vm->tail;
	if (vm->count != 0) {
		while (cur != NULL) {
			if (cur->value == base) {
				target = cur->index;
				break;
			}
			cur = cur->prev;
		}
	}
	if (target != -1) {
		// read the value in the buffer, its entry in page table becomes the most recently used one
		if (cur != vm->tail) { // not the most recent used
			if (cur == vm->head) {
				vm->head = vm->head->next;
				vm->head->prev = NULL;
			}
			else {
				pageNode * before_cur = cur->prev;
				pageNode * after_cur = cur->next;
				before_cur->next = after_cur;
				after_cur->prev = before_cur;
			}
			vm->tail->next = cur;
			cur->prev = vm->tail;
			cur->next = NULL;
			vm->tail = cur;
		}
		return vm->buffer[target*vm->PAGESIZE + offset];
	}
	else { // page fault
		vm->pagefault_num_ptr[0] += 1;
		int head_index = vm->head->index;
		for (int i = 0; i < vm->PAGESIZE; i++) { // swap
			vm->storage[i + vm->head->value * vm->PAGESIZE] = vm->buffer[i + vm->head->index * vm->PAGESIZE];
			vm->buffer[i + vm->head->index * vm->PAGESIZE] = vm->storage[i + base * vm->PAGESIZE];
		}
		pageNode * delete_node = vm->head;
		vm->head = vm->head->next;
		vm->head->prev = NULL;
		delete_node->prev = vm->tail;
		delete_node->next = NULL;
		delete_node->value = base;
		delete_node->index = head_index;
		vm->tail->next = delete_node;
		vm->tail = delete_node;
		return vm->buffer[head_index * vm->PAGESIZE + offset];
	}
  return 123; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
	// page offset length 5, page number 27
	int offset = addr % 32;
	int base = addr >> 5;
	int target = -1;
	pageNode * cur = vm->tail;
	if (vm->count != 0) {
		while (cur != NULL) {
			if (cur->value == base) {
				target = cur->index;
				break;
			}
			cur = cur->prev;
		}
	}
	if (target != -1) {
		vm->buffer[target*vm->PAGESIZE + offset] = value;
		if (cur != vm->tail) { // not the most recent used
			if (cur == vm->head) {
				vm->head = vm->head->next;
				vm->head->prev = NULL;
			}
			else {
				pageNode * before_cur = cur->prev;
				pageNode * after_cur = cur->next;
				before_cur->next = after_cur;
				after_cur->prev = before_cur;
			}
			vm->tail->next = cur;
			cur->prev = vm->tail;
			cur->next = NULL;
			vm->tail = cur;
		}
	}
	else {
		// page fault
		vm->pagefault_num_ptr[0] += 1;
		if (vm->count < vm->PAGE_ENTRIES) { // empty pages exist
			if (vm->count == 0) {
				pageNode * head = new pageNode;
				vm->head = head;
				vm->head->value = base;
				vm->head->index = 0;
				vm->tail = vm->head;
				vm->tail->next = NULL;
				vm->head->prev = NULL;
			}
			else {
				if (vm->count == 1) {
					pageNode * tail = new pageNode;
					tail->value = base;
					tail->index = 1;
					tail->prev = vm->head;
					tail->next = NULL;
					vm->head->next = tail;
					vm->tail = tail;
					vm->head->prev = NULL;
				}
				else {
					pageNode * new_node = new pageNode;
					new_node->value = base;
					new_node->index = vm->count;
					new_node->prev = vm->tail;
					new_node->next = NULL;
					vm->tail->next = new_node;
					vm->tail = new_node;
				}
			}
			vm->buffer[vm->count * vm->PAGESIZE + offset] = value;
			vm->count++;
		}
		else { // page table is full (LRU swap)
			for (int i = 0; i < vm->PAGESIZE; i++) { // swap
				vm->storage[i + vm->head->value * vm->PAGESIZE] = vm->buffer[i + vm->head->index * vm->PAGESIZE];
			}
			int head_index = vm->head->index;
			pageNode * delete_node = vm->head;
			vm->head = vm->head->next;
			vm->head->prev = NULL;
			delete_node->prev = vm->tail;
			delete_node->next = NULL;
			delete_node->value = base;
			delete_node->index = head_index;
			vm->tail->next = delete_node;
			vm->tail = delete_node;
			vm->buffer[head_index*vm->PAGESIZE + offset] = value;
		}
	}
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
	for (int i = 0; i < input_size; i++) {
		results[i] = vm_read(vm, i);
	}
}

