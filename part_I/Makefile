CC=gcc
CXX=g++
CPPFLAGS=-std=c++11
RM=rm -f

SRCS=ref_test.cpp 
OBJS=$(subst .cpp,.o,$(SRCS))

all: ref_test
	./ref_test

ref_test: $(OBJS)
	$(CXX) -o ref_test $(OBJS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $<

clean:
	$(RM) *.o
        

distclean: clean
	$(RM) ref_test
	$(RM) *.o
