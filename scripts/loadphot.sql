begin;
\timing

SET maintenance_work_mem = '4 GB';

create table forcephot_temp as (select * from forcedphotometry);
alter table forcephot_temp alter column id set default nextval('forcedphotometry_id_seq');
alter table forcephot_temp alter column created_at set default now();
alter table forcephot_temp alter column modified set default now();

\copy forcephot_temp (source_id, image_id, flux, fluxerr, flags, ra, dec) from FILENAME with csv header;

alter table forcephot_temp add constraint "imagefk_t" FOREIGN KEY (image_id)
  REFERENCES singleepochsubtractions (id) ON DELETE CASCADE;

alter table forcephot_temp add constraint "sourcefk_t" FOREIGN KEY (source_id)
  REFERENCES sources (id) ON DELETE CASCADE;

create unique index "source_image_t" on forcephot_temp (source_id, image_id);
create index "image_source_t" on forcephot_temp (image_id, source_id);

-- all this happens fast  -- should be invisible to the end user
truncate table forcedphotometry;

-- kill conflicting backends
SELECT pg_terminate_backend(pid) FROM pg_locks WHERE locktype = 'relation' AND relation = (select oid from pg_class where relname = 'forcedphotometry') and pid <> pg_backend_pid();
SELECT pg_terminate_backend(pid) FROM pg_locks WHERE locktype = 'relation' AND relation = (select oid from pg_class where relname = 'sources') and pid <> pg_backend_pid();
SELECT pg_terminate_backend(pid) FROM pg_locks WHERE locktype = 'relation' AND relation = (select oid from pg_class where relname = 'singleepochsubtractions') and pid <> pg_backend_pid();
SELECT pg_sleep(1);
drop table forcedphotometry;

alter table forcephot_temp rename to forcedphotometry;
alter table forcedphotometry rename constraint "imagefk_t" to "imagefk";
alter table forcedphotometry rename constraint "sourcefk_t" to "sourcefk";
alter index "source_image_t" rename to "source_image";
alter index "image_source_t" rename to "image_source";


commit;
