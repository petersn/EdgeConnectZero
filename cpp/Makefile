
CXXFLAGS=-std=c++17 -g -Wall -Wextra -pthread -fPIC -I/usr/local/cuda/include -L/usr/local/cuda/lib -L/usr/local/cuda/lib64
CXXFLAGS+=-O3 -ffast-math -flto `pkg-config --cflags --libs python3`
#LIBS=-lcublas -lcudnn -lcudart

all: self_play_client.so #mcts.so #_edgeconnect_rules.so

#engine_main #fast_server self_play_client

self_play_client: self_play_client.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

self_play_client.so: self_play_client.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $^ $(LIBS)

mcts.so: mcts.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $^ $(LIBS)

fast_server: fast_server.o
	$(CXX) $(CXXFLAGS) -o $@ fast_server.o $(LIBS)

engine_main: engine_main.o fast_eval.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

tests_main: tests_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

edgeconnect_rules_wrap.cxx: edgeconnect_rules.h edgeconnect_rules.i
	swig -c++ -python edgeconnect_rules.i

_edgeconnect_rules.so: edgeconnect_rules_wrap.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $^

%.o: %.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o self_play_client self_play_client.so fast_server engine_main

