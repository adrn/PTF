CREATE TABLE light_curve
(
  pk bigserial NOT NULL,
  objid numeric NOT NULL,
  mjd double precision[] NOT NULL,
  mag double precision[] NOT NULL,
  mag_error double precision[] NOT NULL,
  sys_error double precision[] NOT NULL,
  ra double precision[] NOT NULL,
  "dec" double precision[] NOT NULL,
  flags numeric[] NOT NULL,
  filter_id smallint NOT NULL,
  imaflags numeric[] NOT NULL,
  candidate integer,
  CONSTRAINT light_curve_pk PRIMARY KEY (pk)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE light_curve OWNER TO adrian;

-- Indices:

CREATE INDEX q3c_light_curve_ra_dec_idx
  ON light_curve
  USING btree
  (q3c_ang2ipix(ra, "dec"));

CREATE INDEX light_curve_objid_idx
  ON light_curve
  USING btree
  (objid);
  
CREATE INDEX light_curve_filter_id_idx
  ON light_curve
  USING btree
  (filter_id);
