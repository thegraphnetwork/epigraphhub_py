/* Used for development only */

DROP DATABASE IF EXISTS dev_epigraphhub;
DROP DATABASE IF EXISTS dev_privatehub;
DROP DATABASE IF EXISTS dev_sandbox;
DROP DATABASE IF EXISTS dev_airflow;

DROP ROLE IF EXISTS dev_admin;
DROP ROLE IF EXISTS dev_airflow_user;

CREATE ROLE dev_admin;
ALTER ROLE dev_admin
  WITH SUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS;
ALTER USER dev_admin WITH PASSWORD 'admin';

CREATE ROLE dev_airflow_user;
ALTER ROLE dev_airflow_user
  WITH SUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS;
ALTER USER dev_airflow_user
  WITH PASSWORD 'airflow_password';

DROP ROLE IF EXISTS dev_epigraph;
CREATE ROLE dev_epigraph;
ALTER ROLE dev_epigraph
  WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS;
ALTER USER dev_epigraph
  WITH PASSWORD 'dev_epigraph';

DROP ROLE IF EXISTS dev_external;
CREATE ROLE dev_external;
ALTER ROLE dev_external
  WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB NOLOGIN NOREPLICATION NOBYPASSRLS;
COMMENT ON ROLE dev_external IS 'External analysts with read-only access to  some databases';

CREATE DATABASE dev_epigraphhub OWNER dev_epigraph;
CREATE DATABASE dev_privatehub OWNER dev_epigraph;
CREATE DATABASE dev_sandbox OWNER dev_epigraph;
CREATE DATABASE dev_airflow OWNER dev_airflow_user;

-- Extensions
