CXX=g++8
CXXFLAGS=-std=c++17 -I../include
LDIR=../lib
LIBS=-lmake_transducer

main: main.cpp assignment.o ../part_review/assignment.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(LDIR) $(LIBS)

../part_review/assignment.o: ../part_review/assignment.cpp ../part_review/assignment.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

assignment.o: assignment.cpp assignment.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean cleanest

clean:
	rm -f *.o *~ core

cleanest:
	rm -f *.o *~ core predict.xml model.xml
