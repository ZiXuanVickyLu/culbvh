#include <iostream>
#include "culbvh.cuh"
#include "stacklessbvh.cuh"
int main(int arg, char** args) {
	//culbvh::testLBVH();
	culbvh::testStacklessLBVH();
	return 0;
}