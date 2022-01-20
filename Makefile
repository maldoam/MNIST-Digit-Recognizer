OBJS	= main.o net.o input.o math.o
SOURCE	= main.c net.c input.c math.c
HEADER	= net.h input.h math.h
OUT	= NeuralNet
CC	 = gcc
FLAGS	 = -g -c -Wall
LFLAGS	 = -lm

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

main.o: main.c
	$(CC) $(FLAGS) main.c -std=c99

net.o: net.c
	$(CC) $(FLAGS) net.c -std=c99

input.o: input.c
	$(CC) $(FLAGS) input.c -std=c99

math.o: math.c
	$(CC) $(FLAGS) math.c -std=c99


clean:
	rm -f $(OBJS) $(OUT)

run: $(OUT)
	./$(OUT)