#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <asm/io.h>

#define MAX_USER_SIZE 1024

#define BCM2837_GPIO_ADDRESS 0x3F200000

#define PIN_DAT 17
#define PIN_CLK 18

static struct proc_dir_entry *proc_entry = NULL;
static char buf[MAX_USER_SIZE+1] = {0};
static unsigned int *gpio_registers = NULL;

static void gpio_pin_init(unsigned int pin) {
	unsigned int fsel_bitpos = pin%10;
	unsigned int* gpio_fsel = gpio_registers + (pin/10);
	*gpio_fsel &= ~(7 << (fsel_bitpos*3));
	*gpio_fsel |= (1 << (fsel_bitpos*3));
	return;
}

static void gpio_pin_high(unsigned int pin) {
	unsigned int* gpio_on_register = (unsigned int*)((char*)gpio_registers + 0x1c);
	*gpio_on_register |= (1 << pin);
	return;
}

static void gpio_pin_low(unsigned int pin) {
	unsigned int *gpio_off_register = (unsigned int*)((char*)gpio_registers + 0x28);
	*gpio_off_register |= (1<<pin);
	return;
}

ssize_t driver_read(struct file *file, char __user *user, size_t size, loff_t *off) {
	return copy_to_user(user,"Hello!\n", 7) ? 0 : 7;
}

ssize_t driver_write(struct file *file, const char __user *user, size_t size, loff_t *off) {
	unsigned int i = 0;
	uint32_t d = 0;

	memset(buf, 0x0, sizeof(buf));

	if (size > MAX_USER_SIZE){
		size = MAX_USER_SIZE;
	}

	if (copy_from_user(buf, user, sizeof(uint32_t))) return 0;

	d = (uint32_t)buf[0] |
        (uint32_t)buf[1] << 8 |
        (uint32_t)buf[2] << 16 |
        (uint32_t)buf[3] << 24;

	for(i=0; i < 32; i++){
		gpio_pin_low(PIN_CLK);
		udelay(1);
		if ((d >> i) & 0b1) {
			gpio_pin_high(PIN_DAT); 
		} else {
			gpio_pin_low(PIN_DAT);
		}
		gpio_pin_high(PIN_CLK);
	}

	return size;
}

static const struct proc_ops eye_proc_fops = {
	.proc_read = driver_read,
	.proc_write = driver_write,
};

static int __init panel_driver_init(void) {

	printk("ledpanels driver init: ");
	
	gpio_registers = (int*)ioremap(BCM2837_GPIO_ADDRESS, PAGE_SIZE);

	if (gpio_registers == NULL){
		printk("Failed to map GPIO memory to driver\n");
		return -1;
	}	
	
	// create an entry in the proc-fs
	proc_entry = proc_create("ledpanels", 0666, NULL, &eye_proc_fops);

	if (proc_entry == NULL){
 		printk("Failed to create /proc entry\n");
		return -1;
	}

	gpio_pin_init(PIN_DAT);
	gpio_pin_init(PIN_CLK);

	printk("Successful!\n");

	return 0;
}

static void __exit panel_driver_exit(void) {
	iounmap(gpio_registers);
	proc_remove(proc_entry);
	printk("Ledpanels driver unloaded\n");
	return;
}

module_init(panel_driver_init);
module_exit(panel_driver_exit);

MODULE_LICENSE("MIT");
MODULE_AUTHOR("32");
MODULE_DESCRIPTION("APA212 Matrix driver w. bitbang");
MODULE_VERSION("0");
