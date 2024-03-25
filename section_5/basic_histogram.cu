#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define PATH "./text"

FILE *open_text(const char *file_name);

int main()
{
    // create a text file if it doesn't exist
    FILE *fp = open_text(PATH);
}

FILE *open_text(const char *file_name)
{
    FILE *fp = fopen(file_name, "r");
    if (fp == NULL)
}