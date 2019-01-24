CC=gcc
CXX=g++
CPPFLAGS=-std=c++11
RM=rm -f

SRCS=ref_test.cpp 
OBJS=$(subst .cpp,.o,$(SRCS))
BINDIR=bin

all: ref_test
	$(BINDIR)/ref_test

ref_test: $(OBJS)
	mkdir -p $(BINDIR)
	$(CXX) -o $(BINDIR)/ref_test $(OBJS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $<

clean:
	$(RM) *.o
	$(RM) $(BINDIR)/*        

