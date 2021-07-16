DROP TABLE IF EXISTS articles;
DROP TABLE IF EXISTS fakenews;
DROP TABLE IF EXISTS truenews;
CREATE TABLE fakenews (
    FakeID serial  NOT NULL ,
    title VARCHAR  NOT NULL ,
    text VARCHAR  NOT NULL ,
    subject VARCHAR  NOT NULL ,
    date VARCHAR  NOT NULL ,
    CONSTRAINT PK_fakenews PRIMARY KEY (FakeID)
);
CREATE TABLE truenews (
    TrueID serial  NOT NULL ,
    title VARCHAR  NOT NULL ,
    text VARCHAR  NOT NULL ,
    subject VARCHAR  NOT NULL ,
    date VARCHAR  NOT NULL ,
    CONSTRAINT PK_truenews PRIMARY KEY (TrueID)
);
CREATE TABLE articles (
    ArticlesID serial  NOT NULL ,
    FakeID INT,
    TrueID INT,
    title VARCHAR  NOT NULL ,
    text VARCHAR  NOT NULL ,
    subject VARCHAR  NOT NULL ,
    label INT  NOT NULL ,
    CONSTRAINT PK_articles PRIMARY KEY (ArticlesID)
);
ALTER TABLE articles ADD CONSTRAINT fk_reference_fake_id FOREIGN KEY (FakeID) REFERENCES fakenews(FakeID);
ALTER TABLE articles ADD CONSTRAINT fk_reference_true_id FOREIGN KEY (TrueID) REFERENCES truenews(TrueID);