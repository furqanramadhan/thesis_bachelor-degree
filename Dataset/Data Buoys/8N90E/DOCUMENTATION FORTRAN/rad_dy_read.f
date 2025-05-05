      program rad_dy_read
c
c This program reads daily TAO/TRITON and PIRATA ascii format 
c   shortwave radiation files, for example rad0n110w_dy.ascii. 
c   It creates real time series arrays which are evenly spaced 
c   in time.
c
c   Also created are integer arrays of quality
c
c You can easily adapt this program to your needs.
c
c Programmed by Dai McClurg, NOAA/PMEL/OCRD, August 2000
c  Please email dai@pmel.noaa.gov if you have difficulties.
c
      implicit none
c
      integer nt
      parameter(nt = 20000)
c
      integer n, m
c
      integer nblock, nn, ntime, n1, n2
c
      integer idate(nt), ihms(nt)
      integer iqrad(nt), isrc(nt)
      real deprad, depstd, depmax
c
      real rad(nt), std(nt), max(nt), flag
c
      character infile*80, header*132, aqual*1
c
c .......................................................................
c
      write(*,*) ' Enter the input file name '
      read(*,'(a)') infile
c
      open(1,file=infile,status='old',form='formatted')
c 
c Read total number of hours and blocks of data.
c
      read(1,10) ntime, nblock
   10 format(49x,i5,7x,i3)
c
      write(*,*) ntime, nblock
c
c Read the missing data flag
c
      read(1,20) flag
   20 format(38x,f8.2)
c
      write(*,*) flag
c
c  Initialize data arrays to flag and quality arrays to 5.
c
      do n = 1, nt
        rad(n) = flag
        std(n) = flag
        max(n) = flag
        iqrad(n) = 5
         isrc(n) = 0
      enddo
c
c Read the data.
c
      do m = 1, nblock
        read(1,30) n1, n2, nn
        read(1,50) deprad, depstd, depmax
        read(1,'(a)') header
        do n = n1, n2
          read(1,60) idate(n), ihms(n), rad(n), std(n), max(n), 
     .                aqual , isrc(n)
          if(aqual .eq. 'C') then
            iqrad(n) = -9
          else
            read(aqual,'(i1)') iqrad(n)
          endif
        enddo
      enddo
c
   30 format(51x,i5,4x,i5,1x,i5)
   50 format(14x,3f8.1)
   60 format(1x,i8,1x,i4,3f8.2,1x,a1,1x,i1)
c
      close(1)
c
c Now write out the data and quality arrays to the standard output. 
c
      write(*,*) deprad
c
      do n = 1, ntime
        write(*,70) idate(n), ihms(n), rad(n), std(n), max(n),
     .    iqrad(n), isrc(n)
      enddo
   70 format(x,i8,x,i6,3f8.2,x,i2,x,i1)
c
      end
