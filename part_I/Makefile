CC=gcc
CXX=g++
CPPFLAGS=-std=c++11
RM=rm -f

SRCS=ref_test.cpp 
OBJS=$(subst .cpp,.o,$(SRCS))

all: ref_test
	./bin/ref_test

ref_test: $(OBJS)
	$(CXX) -o ./bin/ref_test $(OBJS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $<

clean:
	$(RM) *.o
	$(RM) ./bin/*        

