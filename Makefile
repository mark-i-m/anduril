
ifeq ($(wildcard /users/markm/kernel-*/kbuild/), )
	KPATH := /usr/src/linux-headers-`uname -r`
else
	KPATH := $(HOME)/kernel-*/kbuild/
endif

obj-m := superultramegafragmentor.o

.PHONY: all clean

all:
	make -C $(KPATH) M=$(CURDIR) modules

clean:
	make -C $(KPATH) M=$(CURDIR) clean
