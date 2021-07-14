-- Creating Database
CREATE DATABASE FakeNewsDetector
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'C'
    LC_CTYPE = 'C'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;
	
-- Creating Tables
DROP TABLE IF EXISTS FakeNews;
CREATE TABLE FakeNews (
     Title VARCHAR,
     Text VARCHAR,
     Subject VARCHAR,
     PublicationDate DATE
);

DROP TABLE IF EXISTS TrueNews;
CREATE TABLE TrueNews (
     Title VARCHAR,
     Text VARCHAR,
     Subject VARCHAR,
     PublicationDate DATE
);

DROP TABLE IF EXISTS Articles;
CREATE TABLE Articles (
     Title VARCHAR NOT NULL,
     Text VARCHAR NOT NULL,
     Subject VARCHAR NOT NULL,
     Label BOOL NOT NULL
);