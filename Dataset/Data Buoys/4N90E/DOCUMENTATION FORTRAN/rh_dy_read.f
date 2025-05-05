      program rh_read
c
c This program reads daily or hourly TAO/TRITON and PIRATA ascii 
c   format rh files for example rh0n110w_dy.ascii. It creates 
c   real time series arrays which are evenly spaced in time
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
      integer iqrh(nt),  isrh(nt)
c
      real rh(nt), deprh
      real flag
c
      integer idy, ihr, i10
c
      character infile*80, header*132, line*132
c
c .......................................................................
c
      write(*,*) ' Enter the input rh file name '
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
   12 format(63x,i7,7x,i3)
c
      write(*,*) ntime, nblock
c
c Read the missing data flag
c
      read(1,20) flag
   20 format(30x,f6.2)
c
      write(*,*) flag
c
c  Initialize data arrays to flag and quality arrays to 5.
c
      do n = 1, nt
          rh(n) = flag
        iqrh(n) = 5
        isrh(n) = 0
      enddo
c
c Read the data.
c
      do m = 1, nblock
        if(idy .ne. 0) then
          read(1,30) n1, n2, nn
          read(1,'(a)') line
          read(line(12:23),*) deprh
        else if(ihr .ne. 0) then
          read(1,31) n1, n2, nn
          read(1,'(a)') line
          read(line(12:23),*) deprh
        else if(i10 .ne. 0) then
          read(1,32) n1, n2, nn
          read(1,'(a)') line
          read(line(12:23),*) deprh
        endif
        read(1,'(a)') header
        if(idy .ne. 0 .or. ihr .ne. 0) then
          do n = n1, n2
            read(1,60) idate(n), ihms(n), rh(n), iqrh(n), isrh(n)
          enddo
        else
          do n = n1, n2
            read(1,62) idate(n), ihms(n), rh(n), iqrh(n), isrh(n)
          enddo
        endif
      enddo
c
   30 format(51x,i5,4x,i5,1x,i5)
   31 format(50x,i7,3x,i7,1x,i7)
   32 format(54x,i7,3x,i7,1x,i7)
c
   60 format(1x,i8,1x,i4,f7.2,1x,i1,1x,i1)
   62 format(1x,i8,1x,i6,f7.2,1x,i1,1x,i1)
c
      close(1)
c
c Now write out the data and quality arrays to the standard output.
c
      write(*,*) deprh
c
      do n = 1, ntime
        write(*,62) idate(n), ihms(n), rh(n), iqrh(n), isrh(n)
      enddo
c
      end
