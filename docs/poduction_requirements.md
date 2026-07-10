# Textbook AI Assistant — Production Requirements

## Initial Production Stage

- Registered users: 100
- Daily active users: 20
- Peak concurrent users: 5
- Expected questions per day: 300
- Normal traffic: Less than 1 request per second
- Burst traffic: 3 requests per second
- Initial textbooks: 1
- Initial vector chunks: 934
- Initial retrieval artifacts: Approximately 4.2 MB

## Growth Stages

| Stage | Registered Users | Concurrent Users | Questions Per Day |
|---|---:|---:|---:|
| MVP | 100 | 5 | 300 |
| Growth | 1,000 | 25 | 3,000 |
| Large | 10,000 | 150 | 30,000 |

## API Requirements

- Maximum question length: 1,000 characters
- Maximum request body: 16 KB
- One textbook per request
- Retrieve top 5 chunks
- Target answer length: 300-700 tokens
- Per-user limit: 10 requests per minute
- Initial daily limit: 100 questions per user

## Performance Targets

- Median response time: Under 7 seconds
- p95 response time: Under 15 seconds
- Maximum backend response target: Under 20 seconds
- FAISS retrieval target: Under 200 milliseconds

## Availability and Recovery

- Availability target: 99.5%
- Recovery Time Objective: 4 hours
- Recovery Point Objective: 24 hours
- S3 artifact versioning: Required
- Deployment rollback: Required
- Multi-region deployment: Not required for MVP

## Security Requirements

- Cognito authentication
- JWT-protected API
- Student, teacher, and administrator roles
- Least-privilege IAM
- S3 Block Public Access
- Encryption at rest and in transit
- No credentials in source code
- No sensitive tokens in logs
- Input validation
- Per-user throttling
- Sanitized error responses

## Cost Requirements

- Soft monthly target: Below $50
- Warning threshold: $50
- Critical budget threshold: $100
- Avoid continuously running compute
- Keep FAISS for the MVP
- Defer OpenSearch and Bedrock Knowledge Bases