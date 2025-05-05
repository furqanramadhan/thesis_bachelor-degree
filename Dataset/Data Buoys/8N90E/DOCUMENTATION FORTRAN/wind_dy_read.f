      program wind_read
c
c This program reads daily or hourly TAO/TRITON and PIRATA ascii 
c   format wind files for example w0n110w_dy.ascii. It creates 
c   real time series arrays which are evenly spaced in time of 
c   zonal and meridional winds (uwnd and vwnd), wind speed, and 
c   wind direction.
c
c Also created are integer arrays of data quality.
c
c You can easily adapt this program to your needs.
c
c Programmed by Dai McClurg, NOAA/PMEL/OCRD,  August 1999. 
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
      integer iqspd(nt), iqdir(nt)
      integer isspd(nt), isdir(nt)
c
      real uwnd(nt), vwnd(nt), spd(nt), dir(nt)
      real flag
c
      real depuwnd, depvwnd, depspd, depdir
      integer idy, ihr
c
      character infile*80, header*132, line*132
c
c .......................................................................
c
      write(*,*) ' Enter the input met file name '
      read(*,'(a)') infile
c
      idy = index(infile, 'dy')
      ihr = index(infile, 'hr')    
c
      open(1,file=infile,status='old',form='formatted')
c
c Read total number of hours and blocks of data and missing data flag
c
      if(idy .ne. 0) then
        read(1,10) ntime, nblock
   10   format(49x,i5,7x,i3)
        write(*,*) ntime, nblock
        read(1,20) flag
   20   format(34x,f6.1)
        write(*,*) flag
      else
        read(1,11) ntime, nblock
   11   format(59x,i7,7x,i3)
        write(*,*) ntime, nblock
        read(1,21) flag
   21   format(34x,f6.1)
        write(*,*) flag
      endif
c
c  Initialize data arrays to flag and quality arrays to 5.
c
      do n = 1, nt
        uwnd(n) = flag
        vwnd(n) = flag
        spd(n)  = flag
        dir(n)  = flag
        iqspd(n)  = 5
        iqdir(n)  = 5
        isspd(n)  = 0
        isdir(n)  = 0
      enddo
c
c Read the data.
c
      do m = 1, nblock
        if(idy .ne. 0) then
          read(1,30) n1, n2, nn
        else
          read(1,31) n1, n2, nn
        endif
        read(1,'(a)') line
        read(line(16:39),*) depuwnd, depvwnd, depspd, depdir
        read(1,'(a)') header
        do n = n1, n2
          read(1,60) idate(n), ihms(n), uwnd(n), vwnd(n), spd(n), 
     .            dir(n), iqspd(n), iqdir(n), isspd(n), isdir(n)
        enddo
      enddo
c
   30 format(51x,i5,4x,i5,1x,i5)
   31 format(50x,i7,3x,i7,1x,i7)
   50 format(14x,4i6)
   60 format(1x,i8,1x,i4,4f6.1,1x,2i1,1x,2i1)
c
      close(1)
c
c Now write out the data and quality arrays to the standard output.
c
      write(*,*) depuwnd, depvwnd, depspd, depdir
c
      do n = 1, ntime
        write(*,60) idate(n), ihms(n), uwnd(n), vwnd(n), spd(n), 
     .           dir(n), iqspd(n), iqdir(n), isspd(n), isdir(n)
      enddo
c
      end
