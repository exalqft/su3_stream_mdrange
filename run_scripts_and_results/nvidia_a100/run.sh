# size of 4D lattices
n_vals=(4 6 8 10 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68)

source ${SRCDIR}/compilation/nvidia_a100/load_modules.sh

echo nt n SUN kernel bw > ${SRCDIR}/run_scripts_and_results/nvidia_a100/results.dat

export CUDA_VISIBLE_DEVICES=0

# 4 GPUs per node, 64 cores per socket
# 1 GPU -> 32 cores
for nt in 32;
do

  export OMP_NUM_THREADS=${nt}
  export OMP_PLACES=cores
  export OMP_PROC_BIND=close

  for n in ${n_vals[@]};
  do

    results=( $(${BUILDDIR}/nvidia_a100/su3xsu3-stream-mdrange-static-mu -n ${n} | grep GB/s | awk '{print $1 " " $2 " " $3}') )
    for i in $(seq 0 5);
    do
      echo ${nt} ${n} "internal" ${results[3*i]} ${results[3*i+1]} >> ${SRCDIR}/run_scripts_and_results/nvidia_a100/results.dat
    done
    for i in $(seq 6 9);
    do
      echo ${nt} ${n} "internal" ${results[3*i]}_${results[3*i+1]} ${results[3*i+2]} >> ${SRCDIR}/run_scripts_and_results/nvidia_a100/results.dat
    done

    results=( $(${BUILDDIR}/nvidia_a100/su3xsu3-stream-mdrange-static-mu-array-SUN -n ${n} | grep GB/s | awk '{print $1 " " $2 " " $3}') )
    for i in $(seq 0 3);
    do
      echo ${nt} ${n} "array" ${results[3*i]} ${results[3*i+1]} >> ${SRCDIR}/run_scripts_and_results/nvidia_a100/results.dat
    done
    for i in $(seq 4 7);
    do
      echo ${nt} ${n} "array" ${results[3*i]}_${results[3*i+1]} ${results[3*i+2]} >> ${SRCDIR}/run_scripts_and_results/nvidia_a100/results.dat
    done

  done
done

    
