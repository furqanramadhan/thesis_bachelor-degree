      program sst_read
c
c This program reads daily, hourly, or 10-minute TAO/TRITON and 
c   PIRATA ascii format sst files for example sst0n110w_dy.ascii. 
c   It creates real time series arrays which are evenly spaced 
c   in time
c
c Also created are integer arrays of data quality.
c
c You can easily adapt this program to your needs.
c
c Programmed by Dai McClurg, NOAA/PMEL/OCRD
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
      integer iqsst(nt), issst(nt)
c
      real sst(nt), depsst
      real flag
c
      integer idy, ihr, i10
c
      character infile*80, header*132, line*132
c
c .......................................................................
c
      write(*,*) ' Enter the input sst file name '
      read(*,'(a)') infile
c
      idy = index(infile, 'dy')
      ihr = index(infile, 'hr')
      i10 = index(infile, '10m')
c
      open(1,file=infile,status='old',form='formatted')
c
c Read total number of hours and blocks of data.
c
      if(idy .ne. 0) then
        read(1,10) ntime, nblock
      else if(ihr .ne. 0) then
        read(1,11) ntime, nblock
      else if(i10 .ne. 0) then
        read(1,12) ntime, nblock
      else
        write(*,*) ' Input file not daily, hourly, or 10 minute. stop'
        stop
      endif
   10 format(49x,i5,7x,i3)
   11 format(59x,i7,7x,i3)
   12 format(63x,i8,7x,i3)
c
      write(*,*) ntime, nblock
c
c Read the missing data flag
c
      if(idy .ne. 0 .or. ihr .ne. 0) then
        read(1,20) flag
   20   format(36x,f6.2)
      else
        read(1,20) flag
   21   format(38x,f7.3)
      endif
c
      write(*,*) flag
c
c  Initialize data arrays to flag and quality arrays to 5.
c
      do n = 1, nt
          sst(n) = flag
        iqsst(n) = 5
        issst(n) = 0
      enddo
c
c Read the data.
c
      do m = 1, nblock
        if(idy .ne. 0) then
          read(1,30) n1, n2, nn
          read(1,'(a)') line
          read(line(12:22),*) depsst
        else if(ihr .ne. 0) then
          read(1,31) n1, n2, nn
          read(1,'(a)') line
          read(line(12:22),*) depsst
        else if(i10 .ne. 0) then
          read(1,32) n1, n2, nn
          read(1,'(a)') line
          read(line(17:23),*) depsst
        endif
        read(1,'(a)') header
        if(idy .ne. 0 .or. ihr .ne. 0) then
          do n = n1, n2
            read(1,60) idate(n), ihms(n), sst(n), iqsst(n)
          enddo
        else
          do n = n1, n2
            read(1,62) idate(n), ihms(n), sst(n), iqsst(n), issst(n)
          enddo
        endif
      enddo
c
   30 format(51x,i5,4x,i5,1x,i5)
   31 format(50x,i7,3x,i7,1x,i7)
   32 format(54x,i8,3x,i8,1x,i8)
c
   60 format(1x,i8,1x,i4,f6.2,1x,i1)
   62 format(1x,i8,1x,i6,f7.3,1x,i1,1x,i1)
c
      close(1)
c
c Now write out the data and quality arrays to the standard output.
c
      write(*,*) depsst
c
      do n = 1, ntime
        write(*,62) idate(n), ihms(n), sst(n), iqsst(n), issst(n)
      enddo
c
      end
