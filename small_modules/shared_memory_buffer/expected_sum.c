#include <stdio.h>

int main() {
    int sum = 0;
    for (int warp = 0; warp < 7; warp++) {
        for (int round = 0; round < 8; round++) {
            int degree = (warp * 1379 + round) % 512;
            for (int i = 0; i < degree; i++) {
                for (int dest_id = 0; dest_id < 4; dest_id++) {
                    if (dest_id == 2) continue;
                    sum += degree + dest_id - i;
                }
            }
        }
    }
    printf("sum = %d\n", sum);
}