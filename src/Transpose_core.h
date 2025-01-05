#ifndef TRANSPOSE_CORE_H
#define TRANSPOSE_CORE_H
#include <iostream>
#include "Mesh.h"

struct Photon3D
{
    Point pos;
    Vec3f dir;
    Real weight = 1.0f;
    Real max_z  = 0;
    Real Ps     = 0;
    int type    = 0;
    bool alive  = true;
    Index curPyramid;
    Index nextPyramid;
    Face nextFace;
    Photon3D(Point pos, Vec3f dir) : pos(pos), dir(dir), nextFace({}, {}, {}) {}
};
enum class CollectType
{
    EMIT       = 0,
    COLLECT    = 1,
    OUTOFRANGE = 2,
    IGNORE     = -1
};
typedef struct resultType
{
    CollectType type = CollectType::IGNORE;
    Index pyramidIndex;
    Point pos;
    Vec3f dir;
    Real weight;
} resultType;
// 由于虚函数表的问题，在cuda上运行的不能直接使用虚函数

class transpose_core
{
    // 单个光子的传输过程，包括发射、传输、吸收、散射、收集
    // 必须只含有POD数据和Kokkos的自有数据结构
    class DefaultEmitCollectStrategy
    {
       public:
        static constexpr int EMIT = 0;
        DefaultEmitCollectStrategy(TetMesh m)
            : mesh(m), emitPyramids("emitPyramids", m.pyramids.extent(0))
        {
            for (Index i = 0; i < mesh.pyramids.extent(0); i++)
            {
                if (GetCollectType(i) == CollectType::EMIT)
                {
                    emitPyramids(emitCount) = i;
                    emitCount++;
                }
            }
        }
        Kokkos::UnorderedMap<Index, CollectType, Kokkos::DefaultExecutionSpace> collect_map;  //
        Kokkos::View<Index*, Kokkos::DefaultExecutionSpace> emitPyramids;
        int emitCount = 0;
        TetMesh mesh;
        CollectType GetCollectType(Index pyIndex) const { return collect_map.value_at(pyIndex); }
        void emit(Photon3D* p)
        {
            // 遍历所有pyramid找到emit类型的
            if (emitCount == 0)
            {
                p->alive = false;
                return;
            }

            // 随机选择一个emit pyramid
            int randIndex         = GetRandomUtils().generate_random_int(0, emitCount - 1);
            Index selectedPyramid = emitPyramids(randIndex);

            // 设置光子初始位置和pyramid
            p->curPyramid = selectedPyramid;
            p->pos        = mesh.points(mesh.pyramids(selectedPyramid).p1);
            //TODO p->dir = 
            p->alive      = true;
        }
    };

   public:
    TetMesh m_mesh;  //根据View的性质，TetMesh实际上是一系列指针
    Photon3D m_photon;
    resultType result;
    DefaultEmitCollectStrategy m_collectStrategy;
    transpose_core(TetMesh mesh)
        : m_mesh(mesh),
          m_collectStrategy(mesh),
          m_photon(Point(0, 0, 0), Vec3f(0, 0, 1))
    {

    }
    void run()
    {
        emit();
        while (m_photon.alive)
        {
            move();
            roulette();
        }
    }
    bool emit()
    {
        // emit a photon
        m_collectStrategy.emit(&m_photon);
        return true;
    }
    bool Get_next_Pyramid(Index* nextPyramid, Real* dist)
    {
        auto& curPyramid = m_photon.curPyramid;
        auto results =
            IntersectionUtils::ray_pyramid_intersection(m_mesh.pyramids[curPyramid], m_photon.pos, m_photon.dir);

        for (int i = 0; i < 4; i++)
        {
            if (results(i).hit && results(i).t > 0 && results(i).t > 1e6 * REALEPS)
            {
                Point intersection_point = m_photon.pos + m_photon.dir * results(i).t;
                Point next_point_pos     = intersection_point + m_photon.dir * 0.1 * m_mesh.GetMinLength();
                *dist                    = results(i).t;

                auto& p1 = results(i).p1;
                auto& p2 = results(i).p2;
                auto& p3 = results(i).p3;
                Face hitFace(p1, p2, p3);

                if (results(i).type == 3)
                {
                    auto adjacentP   = m_mesh.adjacentPyramids.value_at(curPyramid).adjacentPyramids_3;
                    auto adjacentNum = m_mesh.adjacentPyramids.value_at(curPyramid).adjacentCount_3;
                    for (int j = 0; j < adjacentNum; j++)
                    {
                        auto& nextPyramid_ = adjacentP[j];
                        if (m_mesh.pyramids[nextPyramid_].HasFace(hitFace))
                        {
                            *nextPyramid = nextPyramid_;
                            return true;
                        }
                    }
                }
                else if (results(i).type == 2)
                {
                    auto adjacentP   = m_mesh.adjacentPyramids.value_at(curPyramid).adjacentPyramids_2;
                    auto adjacentNum = m_mesh.adjacentPyramids.value_at(curPyramid).adjacentCount_2;
                    for (int j = 0; j < adjacentNum; j++)
                    {
                        auto& nextPyramid_ = adjacentP[j];
                        if (m_mesh.pyramids[nextPyramid_].InPyramid(next_point_pos))
                        {
                            *nextPyramid = nextPyramid_;
                            return true;
                        }
                    }
                }
                else if (results(i).type == 1)
                {
                    auto adjacentP   = m_mesh.adjacentPyramids.value_at(curPyramid).adjacentPyramids_1;
                    auto adjacentNum = m_mesh.adjacentPyramids.value_at(curPyramid).adjacentCount_1;
                    for (int j = 0; j < adjacentNum; j++)
                    {
                        auto& nextPyramid_ = adjacentP[j];
                        if (m_mesh.pyramids[nextPyramid_].InPyramid(next_point_pos))
                        {
                            *nextPyramid = nextPyramid_;
                            return true;
                        }
                    }
                }
                else
                {
                    Kokkos::printf("Error: the type of the intersection is not 1, 2 or 3\n");
                }

                break;
            }  // if (results(i).hit && results(i).t > 0 && results(i).t > REALEPS)
        }
        return false;
    }
    bool move_len(float len)
    {
        m_photon.pos.x += m_photon.dir.x * len;
        m_photon.pos.y += m_photon.dir.y * len;
        m_photon.pos.z += m_photon.dir.z * len;
        return true;
    }
    bool move()
    {
        Real s_;
        const Pyramid::Attribute& cur_Attr = m_mesh.pyramids[m_photon.curPyramid].value;
        const Real& mua                    = cur_Attr.mua;
        const Real& mus                    = cur_Attr.mus;
        const Real& g                      = cur_Attr.g;
        const Real& n                      = cur_Attr.n;
        // move the photon
        if (mua + mus > 0)
        {
            s_ = -log(GetRandomUtils().generate_random_double()) / (mua + mus);
        }
        else
        {
            s_ = 1;
        }
        while (s_ >= 0 && m_photon.alive)
        {
            Real dist = 0;
            Get_next_Pyramid(&m_photon.nextPyramid, &dist);
            auto nowCollectType = m_collectStrategy.GetCollectType(m_photon.nextPyramid);
            switch (nowCollectType)
            {
                case CollectType::COLLECT:
                    m_photon.alive      = false;
                    result.type         = CollectType::COLLECT;
                    result.pyramidIndex = m_photon.nextPyramid;
                    result.pos          = m_photon.pos;
                    result.dir          = m_photon.dir;
                    result.weight       = m_photon.weight;
                    break;
                case CollectType::OUTOFRANGE:
                    m_photon.alive = false;
                    result.type    = CollectType::OUTOFRANGE;
                    break;
                case CollectType::IGNORE: break;
                default: break;
            }

            if (s_ > dist)
            {
                m_photon.Ps += dist;
                move_len(dist);
                s_ -= dist;
                m_photon.pos = m_photon.pos + m_photon.dir * dist;
                DealWithFace();
            }
            else
            {
                move_len(s_);
                m_photon.Ps += s_;
                Absorb(mua, mus);
                Scatter(g);
                s_ = 0;
            }
            m_photon.max_z = m_photon.max_z > m_photon.pos.z ? m_photon.max_z : m_photon.pos.z;
        }
        return true;
    }
    bool roulette()
    {
        if (m_photon.weight < 0.0001)
        {
            if (GetRandomUtils().generate_random_double() > 0.1)
            {
                m_photon.alive = false;
                return false;
            }
            else
            {
                m_photon.weight /= 0.1;
                return true;
            }
        }
        else
        {
            return true;
        }
    }
    void Mirror()
    {
        auto n     = m_photon.nextFace.normal();
        float cdot = n.x * m_photon.dir.x + n.y * m_photon.dir.y + n.z * m_photon.dir.z;
        m_photon.dir.x -= 2.0f * cdot * n.x;
        m_photon.dir.y -= 2.0f * cdot * n.y;
        m_photon.dir.z -= 2.0f * cdot * n.z;
        m_photon.nextPyramid = m_photon.curPyramid;
    }
    void Transmit(float nipnt, float costhi, float costht, Point nor)
    {
        if (costhi > 0.0)
        {
            m_photon.dir.x = nipnt * m_photon.dir.x + (nipnt * costhi - costht) * nor.x;
            m_photon.dir.y = nipnt * m_photon.dir.y + (nipnt * costhi - costht) * nor.y;
            m_photon.dir.z = nipnt * m_photon.dir.z + (nipnt * costhi - costht) * nor.z;
        }
        else
        {
            m_photon.dir.x = nipnt * m_photon.dir.x + (nipnt * costhi + costht) * nor.x;
            m_photon.dir.y = nipnt * m_photon.dir.y + (nipnt * costhi + costht) * nor.y;
            m_photon.dir.z = nipnt * m_photon.dir.z + (nipnt * costhi + costht) * nor.z;
        }
        m_photon.curPyramid = m_photon.nextPyramid;
    }
    void DealWithFace()
    {
        auto nor    = m_photon.nextFace.normal();
        float n     = m_mesh.pyramids[m_photon.curPyramid].value.n;
        float new_n = m_mesh.pyramids[m_photon.nextPyramid].value.n;

        float nipnt = n / new_n;
        if (nipnt == 1)
        {
            m_photon.curPyramid = m_photon.nextPyramid;
            return;
        }
        float costhi = -(m_photon.dir.x * nor.x + m_photon.dir.y * nor.y + m_photon.dir.z * nor.z);
        if (1.0 - powf(nipnt, 2) * (1.0 - powf(costhi, 2)) <= 0.0)
        {
            Mirror();
            return;
        }
        float costht = sqrtf(1.0f - powf(nipnt, 2) * (1.0f - powf(costhi, 2)));
        float thi;

        if (costhi > 0.0)
            thi = acosf(costhi);
        else
            thi = acosf(-costhi);

        float tht = acosf(costht);
        float R;

        if (sinf(thi + tht) <= 1E-15)
            R = powf((nipnt - 1.0f) / (nipnt + 1.0f), 2);
        else
            R = 0.5f * (powf(sinf(thi - tht) / sinf(thi + tht), 2) + powf(tanf(thi - tht) / tanf(thi + tht), 2));

        float xi =  GetRandomUtils().generate_random_double();

        if (xi <= R)
        {
            Mirror();
            return;
        }
        Transmit(nipnt, costhi, costht, nor);
    }
    bool Scatter(float g)
    {
        float xi = 0, theta = 0, phi = 0;
        auto g2 = g * g;

        // Henye-Greenstein scattering
        if (g != 0.0)
        {
            xi = GetRandomUtils().generate_random_double();
            if ((0.0 < xi) && (xi < 1.0))
                theta = (1.0f + g2 - powf((1.0f - g2) / (1.0f - g * (1.0f - 2.0f * xi)), 2)) / (2.0f * g);
            else
                theta = (1.0f - xi) * M_PI;
        }
        else
            theta = 2.0f * GetRandomUtils().generate_random_double() - 1.0f;

        phi             = 2.0f * M_PI * GetRandomUtils().generate_random_double();
        float& cosTheta = theta;
        // Scatter the photon
        float dxn, dyn, dzn;
        float sinTheta = sqrtf(1 - cosTheta * cosTheta);
        float sinPsi   = sinf(phi);
        float cosPsi   = cosf(phi);
        if (fabs(m_photon.dir.z) > 0.999f)
        {
            dxn = sinTheta * cosPsi;
            dyn = sinTheta * sinPsi;
            dzn = m_photon.dir.z * cosTheta / fabs(m_photon.dir.z);
        }
        else
        {
            dxn = sinTheta * (m_photon.dir.x * m_photon.dir.z * cosPsi - m_photon.dir.y * sinPsi) /
                      sqrtf(1.0f - m_photon.dir.z * m_photon.dir.z) +
                  m_photon.dir.x * cosTheta;
            dyn = sinTheta * (m_photon.dir.y * m_photon.dir.z * cosPsi + m_photon.dir.x * sinPsi) /
                      sqrtf(1.0f - m_photon.dir.z * m_photon.dir.z) +
                  m_photon.dir.y * cosTheta;
            dzn = -sinTheta * cosPsi * sqrtf(1.0f - m_photon.dir.z * m_photon.dir.z) + m_photon.dir.z * cosTheta;
        }

        float norm = sqrtf(dxn * dxn + dyn * dyn + dzn * dzn);
        dxn /= norm;
        dyn /= norm;
        dzn /= norm;

        m_photon.dir.x = dxn;
        m_photon.dir.y = dyn;
        m_photon.dir.z = dzn;
        return true;
    }
    bool Absorb(float mua, float mus)
    {
        float dwa = m_photon.weight * mua / (mus + mua);
        m_photon.weight -= dwa;
        return true;
    }
};

#endif