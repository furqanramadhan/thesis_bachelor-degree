
To TAO/TRITON, PIRATA, and RAMA data users:

Please acknowledge the GTMBA Project Office of NOAA/PMEL if you 
use these data in publications.  Also, we would appreciate receiving 
a preprint and/or reprint of publications utilizing the data for 
inclusion in the GTMBA Project bibliography.  Relevant publications
should be sent to:

Global Tropical Moored Buoy Array Project Office
NOAA/Pacific Marine Environmental Laboratory
7600 Sand Point Way NE
Seattle, WA 98115

Comments and questions should be directed to kenneth.connell@noaa.gov

Thank you.

Michael J. McPhaden
GTMBA Project Director
-------------------------------------------------------------------

The following topics will be covered in this readme file:

  1. Winds
  2. Time Stamp
  3. 5-day, Monthly and Quarterly Averages
  4. Sampling, Sensors, and Moorings
  5. Quality Codes
  6. Source Codes
  7. References

1. Winds:

Included in your files are zonal and meridional components
of wind, wind speed, and wind direction.

Winds use the oceanographic convention, so that a wind with 
zonal and meridional components of 1.0 and 1.0, is blowing toward 
the Northeast. Wind components and wind speed have units of meters 
per second while direction has units of degrees, clockwise from
true north. For example, a wind with a direction of 90 degrees
is westerly, i.e., directed to the east. Daily averaged wind 
speeds and directions are based on daily averaged wind velocity
components.

ATLAS buoys measure winds at a height of 4 m above mean sea level 
whereas TRITON buoys measure winds at 3.5 m above mean sea level. 
TRITON buoys replaced ATLAS buoys in the Pacific west of 160E 
beginning in 1999. ATLAS and TRITON wind heights are explicitly 
distinguished in ascii files organized by site. In files containing 
both ATLAS and TRITON data, all winds are given the a height of 
4 meters. (Instrument height in the data files is shown as a 
negative depth).

In August 2011 NDBC Refresh buoys replaced some ATLAS 
buoys in the Pacific. For details on the conversion of 
ATLAS to Refresh, see

  http://www.pmel.noaa.gov/gtmba/refresh-system

The study by Freitag et al ("Calibration procedures and instrumental 
accuracies for ATLAS wind measurements", NOAA. Tech. Memo. OAR 
PMEL-119, 2001) discovered a systematic error in standard and 
NextGeneration ATLAS wind directions of approximately 6.8 degrees 
in the counterclockwise direction. This error was present possibly 
as far back as 1984. Modifications were made to the NextGeneration
ATLAS system in 2000 to correct this error in subsequent deployments, 
and archived NextGeneration ATLAS wind directions were corrected 
(both daily averages and high resolution datasets) on 28 March 2002. 
See

  http://www.pmel.noaa.gov/gtmba/vane-correction

Standard ATLAS wind directions have not been corrected in the archives 
since the exact time when the error began to affect the measurements 
is unknown. Standard ATLAS were used exclusively between 1984 and 1996 
when NextGeneration ATLAS moorings began to replace them. By November 
2001, the standard ATLAS had been phased out and the array was comprised
entirely of NextGeneration systems. Expected RMS error for standard 
ATLAS wind direction is 7.8 degrees (of which 6.8 degrees is a bias) 
while expected RMS error for NextGeneration ATLAS wind directions is 
about +/- 5 degrees with no appreciable bias.
The following wind stations were located on islands

         1n155e, 1s167e, 0n176w, 2n157w

where winds are measured at 10m heights, with the height
at 1s167e (Nauru) changing to 30m after Jan 1 1994.
If you selected any of these island sites, and also chose to
structure your files "by site", then measurement "depth"
will be shown as -10m or -30m in the data files.  If you chose
to structure the data as "one big file", then island data will
be combined with buoy data, all of which will indicate
a measurement "depth" of -4 m.

If you selected daily data at 8n137e or 6s10w, you may
get more than one file per site. This is because the 
original deployments for these sites were at 7n137e and
5s10w, respectively, while present deployments are at 
8n136e and 6s10w. The file names will clearly indicate
which site the data come from. For more details about 
the mooring locations, you can deliver daily average
position data for most deployments under the data category
of "Buoy Positions" on the delivery page.

If you selected the site at 0n110w, you may get data in 
separate files from several groups of deployments clustered 
around 0n110w since mooring locations have been in significantly 
different locations at different times. The file names will 
clearly indicate the site locations. See this page for more
information

  http://www.pmel.noaa.gov/tao/drupal/disdel/110w.html

If you selected high resolution data, you may find that 
you have several files, each with a different averaging
interval, for example, hourly, 15 minute, and 10 minute.
The interval in each file is indicated by one of the 
following file name suffixes: "hr" for hourly, "06h" 
for 6 hour, "02h" for 2 hour, "30m" for 30-minute, and 
similarly for "15m", and "10m". 

In ascii format files, to the right of the data, you will 
find data quality codes for wind speed and direction for
most sites. The exception is for a few inactive sites
which have qualities for uwnd and vwnd components rather
than speed and direction. 

In files organized by site, i.e., delivered after selecting 
"files by site..." in the file structure menu on the delivery 
page, the quality column headers will show "SD" for speed and 
direction qualities, and "UV" for component qualities. 
In 10-minute ascii files by site, you will also you will 
also find source codes. Quality and source codes are defined 
below.

In netcdf files organized by site, you will find quality codes 
with the same shape as the data, and with variable names and 
attributes which give details on which time series the qualities 
are associated with, i.e., speed and direction or zonal and 
meridional components. All but a few inactive sites have 
qualities for speed and direction. These inactive sites 
have qualities associated with zonal and meridional components 
(see below).

If you selected "One big file" in the file structure menu, 
you will see four delivered wind files: zonal wind, meridional 
wind, wind speed, and wind direction. If you selected "Active 
TAO/TRITON", "All Active", or "PIRATA" in the Array menu, then 
only active sites will be included in your delivered files. 
If you selected "All Active and Inactive" there may be a few 
inactive sites included in your files. For zonal and meridional 
wind at all sites except some of the inactive sites (e.g. 0n108w), 
there are no directly associated qualities, because the basic 
measurements, and hence the available qualities, are for wind 
speed and direction. So, in the zonal and meridional wind files, 
the qualities shown all but a few inactive sites, are the poorer 
of the two qualities for speed and direction at the same time and 
site. This is done because the zonal and meridional components 
of wind are computed from the speed and direction at the highest 
available time resolution, and then averaged to daily. At some 
inactive sites, the quality shown is for zonal or meridional 
wind, since the available qualities are for those data, and 
not for wind speed and direction. Similarly, in wind speed and 
direction files, at all but some inactive sites, the quality 
shown is for wind speed or direction. For some of the inactive 
sites which may be in the wind speed and direction files, the 
quality shown is the poorer of the zonal and meridional wind 
qualities since no speed or direction qualities are available.

2. Time Stamp:

Time associated with data that is processed and stored
on board by the moored buoy data acquisition system is
described for each sensor in the sampling protocol web page
at http://www.pmel.noaa.gov/gtmba/sampling.

Time associated with data that have been averaged in
post-processing represents the middle of the averaging
interval. For example, daily means computed from post
processed hourly or 10-min data are averaged over 24
hours starting at 0000 UTC and assigned an observation
time of 1200 UTC.

3. 5-day, Monthly and Quarterly Averages:

If you delivered 5-day, monthly, or quarterly averaged data
these definitions are relevant to your files:

5-Day: Average of data collected during consecutive five day 
intervals. A minimum of 2 daily values are required to compute 
a 5-day average.

Monthly: Average of all the data collected during each month.
A minimum of 15 daily values are required to compute a monthly 
average.

Quarterly: Average of 3 monthly values. A minimum of 2 monthly 
values are required to compute a quarterly average. 12 quarterly 
averages are computed for each year, one for each center month, 
which includes the previous month, the center month, and the 
next month in the average.

4. Sampling, Sensors, and Moorings:

For detailed information about sampling, sensors, and moorings,
see these web pages:

  http://www.pmel.noaa.gov/gtmba/sensor-specifications

  http://www.pmel.noaa.gov/gtmba/sampling

  http://www.pmel.noaa.gov/gtmba/moorings


5. Quality Codes:

In ascii format files organized by site, you will find data 
quality and source codes to the right of the data. In NetCDF 
format files organized by site, you will find quality and 
source variables with the same shape as the data.
These codes are defined below.

Using the quality codes you can tune your analysis to 
trade-off between quality and temporal/spatial coverage.
Quality code definitions are listed below

  0 = Datum Missing.

  1 = Highest Quality. Pre/post-deployment calibrations agree to within
  sensor specifications. In most cases, only pre-deployment calibrations 
  have been applied.

  2 = Default Quality. Default value for sensors presently deployed and 
  for sensors which were either not recovered, not calibratable when 
  recovered, or for which pre-deployment calibrations have been determined 
  to be invalid. In most cases, only pre-deployment calibrations have been 
  applied.

  3 = Adjusted Data. Pre/post calibrations differ, or original data do
  not agree with other data sources (e.g., other in situ data or 
  climatology), or original data are noisy. Data have been adjusted in 
  an attempt to reduce the error.

  4 = Lower Quality. Pre/post calibrations differ, or data do not agree
  with other data sources (e.g., other in situ data or climatology), or 
  data are noisy. Data could not be confidently adjusted to correct 
  for error.

  5 = Sensor or Tube Failed.

6. Source Codes:

  0 - No Sensor, No Data 
  1 - Real Time (Telemetered Mode)
  2 - Derived from Real Time
  3 - Temporally Interpolated from Real Time
  4 - Source Code Inactive at Present
  5 - Recovered from Instrument RAM (Delayed Mode)
  6 - Derived from RAM
  7 - Temporally Interpolated from RAM
  8 - Spatially Interpolated (e.g. vertically) from RAM

7. References:

For more information about TAO/TRITION, PIRATA, and RAMA, see

McPhaden, M.J., A.J. Busalacchi, R. Cheney, J.R. Donguy,K.S. 
Gage, D. Halpern, M. Ji, P. Julian, G. Meyers, G.T. Mitchum, 
P.P. Niiler, J. Picaut, R.W. Reynolds, N. Smith, K. Takeuchi, 
1998: The Tropical Ocean-Global Atmosphere (TOGA) observing 
system:  A decade of progress. J. Geophys. Res., 103, 14,
169-14,240.

Bourles, B., R. Lumpkin, M.J. McPhaden, F. Hernandez, P. Nobre, 
E.Campos, L. Yu, S. Planton, A. Busalacchi, A.D. Moura, J. 
Servain, and J. Trotte, 2008: The PIRATA Program: History, 
Accomplishments, and Future Directions. Bull. Amer. Meteor. 
Soc., 89, 1111-1125.

McPhaden, M.J., G. Meyers, K. Ando, Y. Masumoto, V.S.N. Murty, M.
Ravichandran, F. Syamsudin, J. Vialard, L. Yu, and W. Yu, 2009: RAMA: The
Research Moored Array for African-Asian-Australian Monsoon Analysis and
Prediction. Bull. Am. Meteorol. Soc., 90, 459-480,
doi:10.1175/2008BAMS2608.1
