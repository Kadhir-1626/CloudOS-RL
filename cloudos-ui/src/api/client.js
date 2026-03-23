import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

export const scheduleWorkload = (payload) =>
  api.post('/schedule', payload).then(r => r.data)

export const getStatus = () =>
  api.get('/status').then(r => r.data)

export const getDecisions = (limit = 20) =>
  api.get('/decisions', { params: { limit } }).then(r => r.data)

export const getDecision = (id) =>
  api.get(`/decisions/${id}`).then(r => r.data)

export const explainDecision = (id) =>
  api.post(`/decisions/${id}/explain`).then(r => r.data)