
#KPATH := /usr/src/linux-headers-`uname -r`
KPATH := /users/markm/kernel-*/kbuild/

obj-m := superultramegafragmentor.o

.PHONY: all clean

all:
	make -C $(KPATH) M=$(CURDIR) modules

clean:
	make -C $(KPATH) M=$(CURDIR) clean
