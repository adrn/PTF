CREATE TABLE field
(
  id integer NOT NULL,
  CONSTRAINT field_pk PRIMARY KEY (id)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE field OWNER TO adrian;

CREATE TABLE ccd_exposure
(
  pk serial NOT NULL,
  exp_id bigint NOT NULL,
  mjd double precision NOT NULL,
  field_id integer NOT NULL,
  ccd_id smallint NOT NULL,
  filter_id smallint NOT NULL,
  ra double precision NOT NULL,
  "dec" double precision NOT NULL,
  l double precision NOT NULL,
  b double precision NOT NULL,
  CONSTRAINT ccd_exposure_pk PRIMARY KEY (pk),
  CONSTRAINT field_fk FOREIGN KEY (field_id)
      REFERENCES field (id) MATCH SIMPLE
      ON UPDATE CASCADE ON DELETE CASCADE
)
WITH (
  OIDS=FALSE
);
ALTER TABLE ccd_exposure OWNER TO adrian;

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
  imaflags numeric[] NOT NULL,
  candidate integer,
  CONSTRAINT light_curve_pk PRIMARY KEY (pk)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE light_curve OWNER TO adrian;

CREATE TABLE ccd_exposure_to_light_curve
(
  ccd_exposure_pk integer NOT NULL,
  light_curve_pk bigint NOT NULL,
  CONSTRAINT ccd_exposure_to_light_curve_pk PRIMARY KEY (ccd_exposure_pk, light_curve_pk),
  CONSTRAINT ccd_exposure_fk FOREIGN KEY (ccd_exposure_pk)
      REFERENCES ccd_exposure (pk) MATCH SIMPLE
      ON UPDATE CASCADE ON DELETE CASCADE,
  CONSTRAINT light_curve_fk FOREIGN KEY (light_curve_pk)
      REFERENCES light_curve (pk) MATCH SIMPLE
      ON UPDATE CASCADE ON DELETE CASCADE
)
WITH (
  OIDS=FALSE
);

ALTER TABLE ccd_exposure_to_light_curve OWNER TO adrian;

-- Indices:

CREATE INDEX ccd_exposure_field_id_idx
  ON ccd_exposure
  USING btree
  (field_id);

CREATE INDEX ccd_exposure_ccd_id_idx
  ON ccd_exposure
  USING btree
  (ccd_id);

CREATE INDEX q3c_ccd_exposure_idx
  ON ccd_exposure
  USING btree
  (q3c_ang2ipix(ra, "dec"));

CREATE INDEX q3c_ccd_exposure_lb_idx
  ON ccd_exposure
  USING btree
  (q3c_ang2ipix(l, b));