CXX?=g++
CXXFLAGS?=-O3 -g -std=c++11
INCLUDES?=-I.
LDFLAGS?= -lglog -lgflags

all : nmf nmf2

nmf : nmf.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

nmf2 : nmf2.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

nmf.o : nmf.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

nmf2.o : nmf2.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY:
clean : 
	rm -f nmf.o
	rm -f nmf 
	rm -f nmf2.o
	rm -f nmf2 

