      program rain_dy_read
c
c This program reads daily TAO/TRITON and PIRATA ascii format 
c   rain files, for example r0n110w_dy.ascii. It creates real 
c   time series arrays for rain rate, standard deviation, 
c   and percent time raining, which are evenly spaced in time.
c
c   Also created is an integer array of quality
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
      integer iqrain(nt), isrc(nt)
c
      real rain(nt), std(nt), perc(nt), flag
c
      real deprain, depstd, depperc
c
      character infile*80, header*132
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
   20 format(30x,f6.2)
c
      write(*,*) flag
c
c  Initialize data arrays to flag and quality arrays to 5.
c
      do n = 1, nt
        rain(n) = flag
         std(n) = flag
        perc(n) = flag
        iqrain(n) = 5
          isrc(n) = 0
      enddo
c
c Read the data.
c
      do m = 1, nblock
        read(1,30) n1, n2, nn
        read(1,50) deprain, depstd, depperc
        read(1,'(a)') header
        do n = n1, n2
          read(1,60) idate(n), ihms(n), rain(n), std(n), perc(n),
     .                 iqrain(n), isrc(n)
        enddo
      enddo
c
   30 format(51x,i5,4x,i5,1x,i5)
   50 format(14x,3f6.1)
   60 format(1x,i8,1x,i4,3f6.2,1x,i1,1x,i1)
c
      close(1)
c
c Now write out the data and quality arrays to the standard output. 
c
      write(*,*) deprain
c
      do n = 1, ntime
        write(*,60) idate(n), ihms(n), rain(n), std(n), perc(n), 
     .    iqrain(n), isrc(n)
      enddo
c
      end
