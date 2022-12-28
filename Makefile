
ifeq ($(wildcard /users/markm/kernel-*/kbuild/), )
	KPATH := /usr/src/linux-headers-`uname -r`
else
	KPATH := $(HOME)/kernel-*/kbuild/
endif

obj-m := anduril.o

.PHONY: all clean

KBUILD_CFLAGS += -Wimplicit-fallthrough=3

all:
	make -C $(KPATH) M=$(CURDIR) modules

clean:
	make -C $(KPATH) M=$(CURDIR) clean
