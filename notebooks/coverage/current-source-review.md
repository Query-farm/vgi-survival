# Survival catalog review

Public v0.4.0 actually registers four whole-relation table functions and the
methods view: five logical surfaces. Complete functions/tables/views exports
are retained. No authentication, writers or reference objects were excluded.

Current source0e92f43 has the same TABLE_FUNCTIONS, methods view, argument roles,
return schemas, estimation logic, buffering/transport and dependencies.
The full release-to-current diff adds only examples and metadata in tables.py,
worker.py and examples.py. Source files and dependency declarations are captured.
Current source is reviewed for interface/behavior compatibility, not separately
built or executed. Saved run receipts establish only publicv0.4.0 execution.
Green source CI43582579260 at0e92f43 is reused for notebook-only changes.
