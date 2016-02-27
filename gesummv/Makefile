CC=pgcc
TARGET_GPU=tesla
TARGET_CPU=multicore
CFLAGS=-fast -Minfo=all

gesummv:
	        ${CC} ${CFLAGS} -acc -ta=${TARGET_GPU} gesummv.c -o gpu_gesummv
			        ${CC} ${CFLAGS} -acc -ta=${TARGET_GPU} gesummv_async.c -o gpu_gesummv_async
					        ${CC} ${CFLAGS} -acc -ta=${TARGET_CPU} gesummv.c -o cpu_gesummv

clean:
	        rm -Rf *.o gpu_* cpu_*
