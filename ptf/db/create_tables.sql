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
  filter_id integer[] NOT NULL,
  imaflags numeric[] NOT NULL,
  variability_indices_pk bigint,
  candidate integer,
  CONSTRAINT light_curve_pk PRIMARY KEY (pk),
  CONSTRAINT variability_indices_fk FOREIGN KEY (variability_indices_pk)
      REFERENCES variability_indices (pk) MATCH SIMPLE
      ON UPDATE CASCADE ON DELETE RESTRICT
)
WITH (
  OIDS=FALSE
);
ALTER TABLE light_curve OWNER TO adrian;

CREATE TABLE variability_indices (
    pk bigserial NOT NULL,
    sigma_mu double precision NOT NULL,
    con double precision NOT NULL,
    eta double precision NOT NULL,
    J double precision NOT NULL,
    K double precision NOT NULL,
    CONSTRAINT variability_indices_pk PRIMARY KEY (pk)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE variability_indices OWNER TO adrian;

-- Indices:

CREATE INDEX light_curve_variability_pk_idx
  ON light_curve
  USING btree
  (variability_indices_pk);

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

CREATE INDEX light_curve_candidate_idx
  ON light_curve
  USING btree
  (candidate);
