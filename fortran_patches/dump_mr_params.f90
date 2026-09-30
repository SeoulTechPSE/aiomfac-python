! temporary helper: dump the final MR parameter tables (after MRdata) at full precision.
! Build note: ModMRpart must be compiled with RccTAB and qcca1TAB made public (they are private in the original).
program dump_mr_params
use Mod_kind_param, only : wp
use ModMRpart, only : MRdata, bTABnc, cTABnc, bTABna, cTABna, bTABAC, cTABAC, Cn1TABAC, Cn2TABAC, &
    & omega2TAB, omegaTAB, TABhighestWTF, TABKsp, qcca1TAB, RccTAB
use ModSubgroupProp, only : SMWC, SMWA
implicit none
integer :: u
call MRdata()
open(newunit=u, file='mr_params_dump.txt', status='replace', action='write')
call put2('bTABnc', bTABnc);   call put2('cTABnc', cTABnc)
call put2('bTABna', bTABna);   call put2('cTABna', cTABna)
call put2('bTABAC', bTABAC);   call put2('cTABAC', cTABAC)
call put2('Cn1TABAC', Cn1TABAC); call put2('Cn2TABAC', Cn2TABAC)
call put2('omega2TAB', omega2TAB); call put2('omegaTAB', omegaTAB)
call put2('TABhighestWTF', TABhighestWTF); call put2('TABKsp', TABKsp)
call put2('RccTAB', RccTAB)
write(u,'(A,3(1X,I0))') 'ARRAY qcca1TAB', size(qcca1TAB,1), size(qcca1TAB,2), size(qcca1TAB,3)
write(u,'(*(1X,ES26.17E3))') qcca1TAB
write(u,'(A,2(1X,I0))') 'ARRAY SMWC', size(SMWC), 1
write(u,'(*(1X,ES26.17E3))') SMWC
write(u,'(A,2(1X,I0))') 'ARRAY SMWA', size(SMWA), 1
write(u,'(*(1X,ES26.17E3))') SMWA
close(u)
contains
    subroutine put2(name, a)
    character(len=*), intent(in) :: name
    real(wp), dimension(:,:), intent(in) :: a
    write(u,'(A,A,2(1X,I0))') 'ARRAY ', name, size(a,1), size(a,2)
    write(u,'(*(1X,ES26.17E3))') a
    end subroutine
end program
