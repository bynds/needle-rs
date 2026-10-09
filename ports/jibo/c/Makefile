# The C99 Needle 3 engine. `make` builds for this host; for Jibo:
#   make CC=/path/to/jibo-armcc ARCH="-march=armv7-a -mfpu=neon -mfloat-abi=hard"
# or with the stand-in: make CC=../scripts/jibo-cc-standin.sh (JIBO_SYSROOT set).
CC ?= cc
OUT ?= build
ARCH ?=
WARN = -Wall -Wextra -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wmissing-prototypes -pedantic
CFLAGS ?= -O2
ALL_CFLAGS = -std=c99 -D_POSIX_C_SOURCE=200809L $(WARN) -ffp-contract=off -fno-strict-aliasing $(ARCH) $(CFLAGS)
LDLIBS = -lm -lpthread

SRC = $(wildcard src/*.c)
OBJ = $(SRC:src/%.c=$(OUT)/obj/%.o)
LIBOBJ = $(filter-out $(OUT)/obj/main.o,$(OBJ))

.PHONY: all clean test
all: $(OUT)/needle-jibo-c

$(OUT)/obj/%.o: src/%.c src/*.h
	@mkdir -p $(dir $@)
	$(CC) $(ALL_CFLAGS) -Isrc -c -o $@ $<

$(OUT)/libnd.a: $(LIBOBJ)
	rm -f $@ && ar rcs $@ $^

$(OUT)/needle-jibo-c: $(OUT)/obj/main.o $(OUT)/libnd.a
	$(CC) $(ALL_CFLAGS) -o $@ $^ $(LDLIBS)

$(OUT)/test_%: tests/test_%.c $(OUT)/libnd.a
	$(CC) $(ALL_CFLAGS) -Isrc -o $@ $^ $(LDLIBS)

# Benchmarks (tests/bench_*.c); per-operation numbers need CFLAGS="-O2 -DND_PROFILE".
$(OUT)/bench_%: tests/bench_%.c $(OUT)/libnd.a
	$(CC) $(ALL_CFLAGS) -Isrc -o $@ $^ $(LDLIBS)

clean:
	rm -rf $(OUT)
