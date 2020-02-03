
CXXFLAGS=-std=c++17 -g -pthread -fPIC -I/usr/local/cuda/include -L/usr/local/cuda/lib -L/usr/local/cuda/lib64
#CXXFLAGS+=-Wall -Wextra
CXXFLAGS+=-O3 -ffast-math -flto `pkg-config --cflags --libs python3`
#LIBS=-lcublas -lcudnn -lcudart

CXXFLAGS+=-I/nix/store/2hvzi3lhfr1k65r2pc9iy1n8fk647zf1-python3.7-tensorflow-gpu-1.14.0/lib/python3.7/site-packages/tensorflow/include -fPIC
#CXXFLAGS+=-I
#CXXFLAGS+=-ltensorflow -ltensorflow_framework

TFLIBS=
TFLIBS+=/nix/store/1jai706l893rlkjkhhgp2f1970v341fq-tensorflow-gpu-1.14.0/lib/libtensorflow.so
TFLIBS+=/nix/store/1jai706l893rlkjkhhgp2f1970v341fq-tensorflow-gpu-1.14.0/lib/libtensorflow_framework.so

all: self_play_client.so mcts.so #use_tensorflow #_edgeconnect_rules.so

#engine_main #fast_server self_play_client

self_play_client: self_play_client.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

self_play_client.so: self_play_client.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $^ $(LIBS)

mcts.so: mcts.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$@ -o $@ $^ $(LIBS) $(TFLIBS)

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

use_tensorflow: use_tensorflow.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(TFLIBS)

%.o: %.cxx
	$(CXX) -c $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o self_play_client self_play_client.so fast_server engine_main

