! temporary helper: dump the final SR parameter arrays used by the model (full precision)
program dump_sr_params
use Mod_kind_param, only : wp
use ModSystemProp, only : Nmaingroups, topsubno
use ModSRparam, only : SRdata, ARR, BRR, CRR, SR_RR, SR_QQ
use ModSubgroupProp, only : NKTAB, GroupMW, Ioncharge
implicit none
integer :: u, i, j
call SRdata()
open(newunit=u, file='sr_params_dump.txt', status='replace', action='write')
write(u,'(A,2(1X,I0))') 'DIMS Nmaingroups topsubno', Nmaingroups, topsubno
write(u,'(A,*(1X,ES26.17E3))') 'SR_RR', SR_RR
write(u,'(A,*(1X,ES26.17E3))') 'SR_QQ', SR_QQ
do j = 1, Nmaingroups
    write(u,'(A,I0,*(1X,ES26.17E3))') 'ARRcol', j, (ARR(i,j), i=1,Nmaingroups)
enddo
do j = 1, Nmaingroups
    write(u,'(A,I0,*(1X,ES26.17E3))') 'BRRcol', j, (BRR(i,j), i=1,Nmaingroups)
enddo
do j = 1, Nmaingroups
    write(u,'(A,I0,*(1X,ES26.17E3))') 'CRRcol', j, (CRR(i,j), i=1,Nmaingroups)
enddo
write(u,'(A,*(1X,I0))') 'NKTAB', NKTAB
write(u,'(A,*(1X,ES26.17E3))') 'GroupMW', GroupMW
write(u,'(A,*(1X,I0))') 'Ioncharge', Ioncharge
close(u)
end program
