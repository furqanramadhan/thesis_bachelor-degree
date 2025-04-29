      program t_dy_read
c
c This program reads TAO/TRITON and PIRATA ascii-format temperature
c   files, for example t15n38w_dy.ascii. It creates an array called 
c   t, which is evenly spaced in time, and an array called iqual
c   which contains the data quality for each depth.
c
c You can easily adapt this program to your needs.
c
c Programmed by Dai McClurg, NOAA/PMEL/OCRD, August 1999
c
      implicit none
c
      integer nz, nt
      parameter(nz = 42, nt = 20000)
c
      integer k, n, m
c
      integer nblock, nk, ndep, nn, ntime, n1, n2
c
      integer kdep(nz)
      integer iqual(nz,nt), isrc(nz,nt), idate(nt), ihms(nt)
c
      real flag, depth(nz), t(nz,nt)
c
      character infile*80, header*132, depline*256, frmt*160
c
c .......................................................................
c
      write(*,*) ' Enter the input temperature file name'
      read(*,'(a)') infile
c
      open(1,file=infile,status='old',form='formatted')
c 
c Read total number of days, depths and blocks of data.
c
      read(1,10) ntime, ndep, nblock
   10 format(49x,i5,7x,i3,8x,i3)
c
      write(*,*) ntime, ndep, nblock
c
c Read the missing data flag
c
      read(1,20) flag
c
   20 format(40x,f7.3)
c
      write(*,*) flag
c
c  Initialize t array to flag and iqual array to 5.
c
      do k = 1, nz
        do n = 1, nt
          t(k,n) = flag
          iqual(k,n) = 5
           isrc(k,n) = 0
        enddo
      enddo
c
c Read the data
c
      do m = 1, nblock
        read(1,30) n1, n2, nn, nk
        call blank(frmt)
        write(frmt,140) nk
        read(1,frmt) (kdep(k),k=1,nk)
        read(1,'(a)') depline
        read(depline(15:),*) (depth(kdep(k)),k=1,nk)
        read(1,'(a)') header
        call blank(frmt)
        write(frmt,160) nk, nk, nk
        do n = n1, n2
          read(1,frmt) idate(n), ihms(n), (t(kdep(k),n),k=1,nk), 
     .      (iqual(kdep(k),n),k=1,nk), (isrc(kdep(k),n),k=1,nk)
        enddo
      enddo
c
   30 format(50x,i6,3x,i6,1x,i6,7x,i3)
  140 format('(15x,',i3,'i7)')
  160 format('(1x,i8,1x,i4,1x,',i3,'f7.3,1x,',i3,'i1,1x,',i3,'i1)')
c
      close(1)
c
c Write out the depth, data, and quality arrays to the 
c   standard output. 
c
      write(*,*) 'depth = ', (depth(k),k=1,ndep)
c
c For some files this statement may be too long for your max output
c   record length on your terminal. If so, comment out these lines.
c
      nk = ndep
      call blank(frmt)
      write(frmt,70) ndep, ndep, ndep
c
      do n = 1, ntime
        write(*,frmt) idate(n), ihms(n),(t(k,n),k=1,ndep),
     .       (iqual(k,n),k=1,ndep),  (isrc(k,n),k=1,ndep), n
      enddo
c
   70 format('(1x,i8,1x,i4,1x,',i3,'f7.3,1x,',i3,'i1,1x,',i3,'i1,i7)')
c
      end
c
c.............................................................
c
      subroutine blank(string)
c
c blank out the string from 1 to its declared length
c
      character*(*) string
c
      integer i
c
      do i = 1, len(string)
        string(i:i) = ' '
      enddo
c
      return
      end
