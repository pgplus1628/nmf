CXX?=g++
CXXFLAGS?=-O3 -g -std=c++11
INCLUDES?=-I.
LDFLAGS?= -lglog -lgflags

nmf : nmf.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

nmf.o : nmf.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


.PHONY:
clean : 
	rm -f nmf.o
	rm -f nmf 
