include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

saintvenant: saintvenant.o
	-${CLINKER} -o saintvenant saintvenant.o  ${PETSC_LIB}
	${RM} saintvenant.o


